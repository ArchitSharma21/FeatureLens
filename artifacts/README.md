# Generated artifacts

This directory intentionally ships without invented empirical results.

Run the full study on a CUDA machine:

```bash
python experiments/run_all.py
```

For an interruptible session:

```bash
python experiments/run_all.py --resume
```

If the expensive activation and causal artifacts already exist, regenerate only CPU analysis/report outputs with:

```bash
python experiments/run_analysis_only.py
```

v0.14 produces:

- prompt-wide activation caches plus separate final-token sparse activations;
- `feature_catalog.csv`;
- `layer_metrics.csv`;
- `stability.csv`;
- `selection_stability.csv`;
- `causal_results.csv`;
- `feature_set_results.csv`;
- `study_feature_summary.csv`;
- `study_summary.json`;
- `summary.json`;
- `report.md`;
- report figures including association-vs-causality and candidate-stability plots.

Validate the public artifact set with:

```bash
python -m scripts.validate_artifacts
```

`artifacts/activations/` contains large residual/sparse caches and stays gitignored. Commit only the small CSV/JSON/report/figure outputs if you want the live **Offline study** tab to display measured results and benchmark-derived feature hints.
