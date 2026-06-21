# Paper Manuscript

Publication-ready manuscript for the CMS provider fraud detection study.

## Files

| File | Section |
|------|---------|
| [`01_abstract.md`](01_abstract.md) | Abstract |
| [`02_introduction.md`](02_introduction.md) | §1 Introduction |
| [`03_literature_review.md`](03_literature_review.md) | §2 Literature Review + §3 Research Gap |
| [`04_methodology.md`](04_methodology.md) | §4 Methodology |
| [`05_experimental_setup.md`](05_experimental_setup.md) | §5 Experimental Setup |
| [`06_results.md`](06_results.md) | §6 Results + §7 Ablations |
| [`07_discussion.md`](07_discussion.md) | §8 Discussion + Threats + Future Work |
| [`08_conclusion.md`](08_conclusion.md) | §9 Conclusion |
| [`09_appendix.md`](09_appendix.md) | Appendices A–E |
| [`full_manuscript.md`](full_manuscript.md) | **Complete concatenated manuscript** |

## Key result (verified)

**Logistic regression** achieves the highest mean AUPRC (**0.6810 ± 0.0389**). GraphSAGE (0.6530), R-GCN (0.6542), and best fusion (0.6671) do not beat this baseline under the locked evaluation protocol.

## Figures

See [`../docs/figure_catalog.md`](../docs/figure_catalog.md) and `../figures/pdf/`.

## Reproduction

See [`../docs/reproducibility.md`](../docs/reproducibility.md) and [`../artifacts/published/MANIFEST.json`](../artifacts/published/MANIFEST.json).
