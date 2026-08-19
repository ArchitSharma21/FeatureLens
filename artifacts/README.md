# Generated artifacts

The repository ships without invented empirical results. The finalized v0.16 study uses prompt-wide feature evidence plus two causal position policies.

Fresh full study:

```bash
python -m experiments.run_all --resume
```

Upgrade an already completed v0.15 study without recollecting discovery activations:

```bash
python -m experiments.run_causal_addendum --resume
```

The public artifact set includes:

- `feature_catalog.csv`;
- `layer_metrics.csv`;
- `stability.csv`;
- `selection_stability.csv`;
- `causal_results_final_token.csv`;
- `causal_results_max_active.csv`;
- `causal_position_summary.csv`;
- `feature_set_results.csv`;
- `study_feature_summary.csv`;
- `study_summary.json`;
- `summary.json`;
- `report.md`;
- report figures including causal-position sensitivity and association-vs-causality.

Validate before commit:

```bash
python -m scripts.validate_artifacts
```

`artifacts/activations/` remains gitignored. Commit only the small CSV/JSON/report/figure outputs.
