# Offline study artifacts

This directory contains the small, publishable outputs from the completed FeatureLens study. Large activation matrices, model weights, SAE checkpoints, and task checkpoint markers are intentionally excluded.

Primary study artifacts:

- `feature_catalog.csv` — train-selected SAE candidates and held-out AUROC/F1.
- `layer_metrics.csv` — dense-probe and SAE reconstruction diagnostics.
- `stability.csv` — paraphrase stability measurements.
- `selection_stability.csv` — 128-resample candidate-selection sensitivity.
- `causal_results_final_token.csv` — final-token causal baseline.
- `causal_results_max_active.csv` — max-feature-activation causal study.
- `causal_position_summary.csv` — coverage and task-level position-sensitivity synthesis.
- `feature_set_results.csv` — final-token top-1/3/5 feature-set diagnostic.
- `study_feature_summary.csv` / `study_summary.json` — cross-concept evidence synthesis.
- `summary.json` / `report.md` — measured executive summary and report.
- `split.json` — fixed train/held-out paraphrase-group split.
- `figures/` — report figures used by the public Study tab.

Run `python -m scripts.validate_artifacts` to validate the committed study bundle.
