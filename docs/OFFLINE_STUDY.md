# FeatureLens offline study

## Goal

The offline study answers the project question at dataset scale:

> Do sparse features that predict a controlled concept on held-out prompts also produce behaviorally specific causal effects?

The live app is exploratory. The offline study is where FeatureLens makes held-out, random-controlled, uncertainty-aware claims.

## Representation choice

v0.14 uses **prompt-wide max-pooled SAE activation** for concept evidence. For each prompt and SAE feature, the stored value is the maximum activation across non-padding prompt tokens. This change is motivated by the live finding that a strong final-token activation can reflect a lexical cue rather than the prompt's semantic concept.

The collector also saves `features_final_layer{layer}.npz` so local final-token analyses remain reproducible. Dense residual linear probes continue to use the final prompt-token residual and are therefore reported as a separate baseline rather than as an identical pooling scheme.

## Full pipeline

```bash
python -m experiments.run_all
```

Stages:

1. `build_dataset` — materialize 224 discovery prompts and 28 causal tasks.
2. `collect_activations` — Qwen3 residual capture; prompt-wide and final-token SAE feature artifacts.
3. `evaluate_features` — grouped train/test split, train-only feature selection, held-out AUROC/F1, dense residual probe, paraphrase stability.
4. `run_causal` — selected-feature ablation/amplification with exact continuation scoring and norm-matched random ensembles.
5. `run_feature_sets` — top-1/3/5 joint ablations and random controls.
6. `analyze_stability` — 128 balanced activation resamples from saved prompt-wide features.
7. `analyze_study` — join predictive, robustness, stability, and random-normalized causal evidence by concept.
8. `make_report` — measured figures, summary JSON, and narrative report.
9. `validate_artifacts` — schema/completeness guard before committing public results.

## Resume after interruption

```bash
python -m experiments.run_all --resume
```

The runner checks expected stage outputs and skips completed stages. This is intended for preemptible or quota-limited GPU sessions.

## CPU-only re-analysis

After activation and causal inference artifacts exist:

```bash
python -m experiments.run_analysis_only
```

This reruns feature evaluation, candidate stability, study synthesis, figures, report generation, and artifact validation without another model forward pass.

## Main outputs

`study_feature_summary.csv` contains one selected feature per controlled concept with:

- held-out AUROC and F1;
- training activation rates;
- activation-resample selection support;
- paraphrase TopK Jaccard and sparse cosine;
- causal-task feature-active rate;
- mean absolute and signed target effect;
- norm-matched random mean absolute target effect;
- target-specificity ratio, paired advantage, bootstrap CI, sign-flip p-value;
- equivalent next-token JS specificity metrics.

`study_summary.json` adds descriptive cross-concept Spearman correlations such as held-out AUROC versus target-specificity ratio. There are only seven controlled concepts, so these correlations are **descriptive**, not significance claims.

## Interpretation guardrails

- Prompt-wide max pooling detects whether a feature appears anywhere in the prompt; it discards token order.
- A high held-out AUROC remains correlational evidence.
- Candidate resample support measures shortlist sensitivity under the configured activation-resampling scheme, not feature truth or semantic purity.
- A target-specificity ratio above 1 means the SAE edit moved the exact target more than the mean norm-matched random edit; uncertainty and task coverage still matter.
- JS specificity asks a different question from target specificity: a feature can reshape the local distribution without specifically controlling the chosen target.
- Cross-concept correlations have n=7 and are descriptive.
- Large activation caches should not be committed to the repository.

## Colab workflow

A ready-to-run Colab notebook is included at:

`notebooks/FeatureLens_Offline_Study_Colab.ipynb`

See [`docs/COLAB.md`](COLAB.md) for the persistence model and post-run artifact workflow. The notebook mounts Google Drive for `artifacts/`, keeps the Hugging Face model cache on the local Colab VM, selects a conservative activation batch from available VRAM, and runs the pipeline with `--resume`.

v0.15 additionally checkpoints `run_causal` and `run_feature_sets` after each completed task. If a runtime ends mid-stage, the next `--resume` run skips complete tasks within that stage rather than repeating the entire causal or feature-set benchmark.

The full runner also accepts:

```bash
python -m experiments.run_all --resume --activation-batch-size 8 --activation-max-length 192
```

These two activation flags only affect memory/time during activation collection.
