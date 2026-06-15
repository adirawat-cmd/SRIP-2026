"""
R-GCN training loop with neighbor sampling, early stopping, and tracking.
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor, nn
from torch_geometric.loader import NeighborLoader

from hgad_cms.evaluation.metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
    compute_confusion_matrix,
)
from hgad_cms.exceptions import GNNError
from hgad_cms.graph.constants import NODE_PROVIDER
from hgad_cms.rgcn.config import RGCNConfig, _default_fanout
from hgad_cms.rgcn.inference import FoldGraphData, predict_provider_scores
from hgad_cms.rgcn.model import HeteroRGCN
from hgad_cms.tracking.experiment import ExperimentTracker

logger = logging.getLogger(__name__)


@dataclass
class FoldTrainResult:
    fold_id: int
    metrics: ClassificationMetrics
    confusion_matrix: list[list[int]]
    best_epoch: int
    best_val_auprc: float
    history: list[dict[str, float]] = field(default_factory=list)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str | None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _labeled_train_indices(fold_data: FoldGraphData) -> Tensor:
    labels = fold_data.train_data[NODE_PROVIDER].y.cpu()
    train_idx = fold_data.train_provider_indices.cpu()
    train_labels = labels[train_idx]
    labeled_mask = train_labels >= 0
    if not bool(labeled_mask.any()):
        raise GNNError("No labeled train providers found")
    return train_idx[labeled_mask]


def _positive_class_weight(fold_data: FoldGraphData) -> float:
    labels = fold_data.train_data[NODE_PROVIDER].y.detach().cpu().numpy()
    labeled = labels[labels >= 0]
    positives = float((labeled == 1).sum())
    negatives = float((labeled == 0).sum())
    if positives <= 0:
        return 1.0
    return max(1.0, negatives / positives)


def _build_neighbor_loader(
    data,
    input_nodes: Tensor,
    config: RGCNConfig,
    *,
    shuffle: bool,
) -> NeighborLoader:
    fanout = list(config.fanout[: config.num_layers])
    return NeighborLoader(
        data,
        num_neighbors=fanout,
        input_nodes=(NODE_PROVIDER, input_nodes),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


class RGCNTrainer:
    """Train heterogeneous R-GCN with mini-batch neighbor sampling."""

    def __init__(
        self,
        config: RGCNConfig,
        *,
        device: str | None = "auto",
        tracker: ExperimentTracker | None = None,
    ) -> None:
        self.config = config
        self.device = _resolve_device(device)
        self.tracker = tracker
        self.last_model: HeteroRGCN | None = None

    def _in_channels(self, fold_data: FoldGraphData) -> dict[str, int]:
        return {
            node_type: fold_data.train_data[node_type].x.size(-1)
            for node_type in fold_data.train_data.node_types
        }

    def _make_model(self, fold_data: FoldGraphData) -> HeteroRGCN:
        metadata = fold_data.train_data.metadata()
        return HeteroRGCN(
            metadata,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            num_bases=self.config.num_bases,
            relation_ablation=self.config.relation_ablation,
            in_channels=self._in_channels(fold_data),
        ).to(self.device)

    def _train_epoch(
        self,
        model: HeteroRGCN,
        loader: NeighborLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        model.train()
        total_loss = 0.0
        total_steps = 0
        for batch in loader:
            batch = batch.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.x_dict, batch.edge_index_dict)[NODE_PROVIDER]
            provider_batch = batch[NODE_PROVIDER]
            seed_count = provider_batch.batch_size
            seed_logits = logits[:seed_count]
            seed_labels = provider_batch.y[:seed_count]
            labeled = seed_labels >= 0
            if not bool(labeled.any()):
                continue
            loss = criterion(seed_logits[labeled], seed_labels[labeled])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_steps += 1
        return total_loss / max(total_steps, 1)

    @torch.no_grad()
    def _evaluate(
        self,
        model: HeteroRGCN,
        fold_data: FoldGraphData,
    ) -> tuple[ClassificationMetrics, list[list[int]]]:
        scores = predict_provider_scores(
            model,
            fold_data.inference_data,
            fold_data.val_provider_indices,
            device=self.device,
        )
        metrics = compute_classification_metrics(fold_data.val_labels, scores)
        cm = compute_confusion_matrix(fold_data.val_labels, scores).tolist()
        return metrics, cm

    def train_fold(
        self,
        fold_data: FoldGraphData,
        *,
        fold_id: int,
        run_id: str,
    ) -> FoldTrainResult:
        _set_seed(self.config.seed + fold_id)
        model = self._make_model(fold_data)
        train_data = fold_data.train_data.clone().to(self.device)

        labeled_indices = _labeled_train_indices(fold_data)
        train_loader = _build_neighbor_loader(
            train_data,
            labeled_indices,
            self.config,
            shuffle=True,
        )

        pos_weight = torch.tensor([_positive_class_weight(fold_data)], device=self.device)
        criterion: nn.Module = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        best_state: dict[str, Tensor] | None = None
        best_auprc = -1.0
        best_epoch = 0
        epochs_without_improve = 0
        history: list[dict[str, float]] = []

        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self._train_epoch(model, train_loader, optimizer, criterion)
            metrics, _ = self._evaluate(model, fold_data)
            val_auprc = metrics.auprc
            history.append({"epoch": float(epoch), "train_loss": train_loss, "val_auprc": val_auprc})

            if self.tracker is not None:
                self.tracker.log_epoch(
                    run_id=run_id,
                    fold_id=fold_id,
                    config=self.config,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_auprc=val_auprc,
                )

            logger.info(
                "fold=%s epoch=%s loss=%.4f val_auprc=%.4f",
                fold_id,
                epoch,
                train_loss,
                val_auprc,
            )

            if val_auprc > best_auprc + self.config.min_delta:
                best_auprc = val_auprc
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= self.config.patience:
                    logger.info("Early stopping fold=%s at epoch=%s", fold_id, epoch)
                    break

        if best_state is None:
            raise GNNError(f"Training produced no valid checkpoint for fold {fold_id}")

        model.load_state_dict(best_state)
        final_metrics, cm = self._evaluate(model, fold_data)
        self.last_model = model
        return FoldTrainResult(
            fold_id=fold_id,
            metrics=final_metrics,
            confusion_matrix=cm,
            best_epoch=best_epoch,
            best_val_auprc=best_auprc,
            history=history,
        )
