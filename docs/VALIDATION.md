# FeatureLens v0.16 validation

v0.16 changes the **offline causal methodology**, not the already validated live HF inference UI. Do not spend ZeroGPU quota retesting live Workbench paths.

## Local software gate

```bash
python3 -m pytest -q
python3 -m compileall -q app.py featurelens experiments scripts
python3 -m ruff check app.py featurelens experiments tests scripts
python3 scripts/ui_smoke.py
python3 scripts/release_check.py
```

## Addendum acceptance

Use `notebooks/FeatureLens_Causal_Addendum_Colab.ipynb` with the completed v0.15 Drive run.

The addendum is successful when:

1. `causal_results_final_token.csv` exists and preserves the v0.15 baseline.
2. `causal_results_max_active.csv` completes all 28 causal tasks with 8 random controls per intervention.
3. `causal_position_summary.csv` contains both `final_token` and `max_feature_activation` policies.
4. `study_summary.json` declares `max_feature_activation` as the primary causal position policy and causal-task-level inference as the statistical unit.
5. `python -m scripts.validate_artifacts` prints `PASS`.
6. `FeatureLens_offline_results_v016.zip` is created without activation caches.

## HF acceptance after final artifacts are committed

No GPU call is required. Open **Study** and verify:

- the measured headline is populated;
- final-token and max-active coverage/specificity are visible;
- the causal-position table and figure render;
- the association-vs-causality figure uses max-active specificity;
- no placeholder or old v0.15 significance language remains.
