# FeatureLens final validation

Run the complete local software gate before publishing:

```bash
python3 -m pytest -q
python3 -m compileall -q app.py featurelens experiments scripts
python3 -m ruff check app.py featurelens experiments tests scripts
python3 scripts/ui_smoke.py
python3 scripts/release_check.py
python3 -m scripts.validate_artifacts
```

## Public study checks

The committed `artifacts/` bundle must:

1. contain both `causal_results_final_token.csv` and `causal_results_max_active.csv`;
2. use `max_feature_activation` as the primary causal policy in `study_summary.json`;
3. document causal-task-level paired inference;
4. report 224 discovery prompts and 28 causal tasks;
5. include the six report figures required by `scripts.validate_artifacts`;
6. contain no activation matrices, model weights, SAE checkpoints, or completion markers.

## HF Space acceptance

No new model inference needs to be rerun for the final publication if the software checks pass. Verify visually that:

- the **Study** tab loads measured results rather than the empty-state message;
- the measured headline and causal-position comparison are visible;
- the Workbench and other previously validated live paths still render;
- the interface remains version-neutral and follows `DESIGN.md`.

## Reproducibility

For a fresh study, use `notebooks/FeatureLens_Offline_Study_Colab.ipynb` or:

```bash
python -m experiments.run_all --resume
```

The causal-addendum notebook is retained only as a migration/reproduction utility for an already-completed final-token baseline.
