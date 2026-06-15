"""Project-specific exceptions."""


class HGADError(Exception):
    """Base exception for hgad_cms."""


class DataLoadError(HGADError):
    """Raised when raw CMS files cannot be loaded."""


class DataValidationError(HGADError):
    """Raised when processed data fails G1 validation checks."""


class DataMergeError(HGADError):
    """Raised when claim or label merging fails."""


class DataCleanError(HGADError):
    """Raised when cleaning transforms fail."""


class GraphBuildError(HGADError):
    """Raised when heterogeneous graph construction fails."""


class GraphValidationError(HGADError):
    """Raised when Gate G2 graph validation fails."""


class SplitError(HGADError):
    """Raised when CV split creation or loading fails."""


class BaselineError(HGADError):
    """Raised when baseline model training or inference fails."""


class EvaluationError(HGADError):
    """Raised when evaluation or Gate G3/G4 validation fails."""


class GNNError(HGADError):
    """Raised when GNN training or inference fails."""


class FusionError(HGADError):
    """Raised when hybrid fusion training or evaluation fails."""
