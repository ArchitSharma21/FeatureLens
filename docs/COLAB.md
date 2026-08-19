# Running the FeatureLens offline study on Google Colab

This guide is for the **offline empirical study**, not the public Hugging Face Space. The Space remains the interactive demo; Colab is only used to produce the study artifacts once.

The easiest route is the included notebook:

`notebooks/FeatureLens_Offline_Study_Colab.ipynb`

## What the run does

The full study executes:

1. build/verify the 224-prompt discovery set and 28 causal tasks;
2. collect Qwen3 residuals and prompt-wide Qwen-Scope SAE activations at layers 4, 14, and 26;
3. fit/evaluate held-out SAE-feature classifiers and the dense residual probe;
4. run single-feature random-controlled causal interventions;
5. run 1/3/5 feature-set interventions and controls;
6. compute candidate-selection stability, study synthesis, figures, and report;
7. validate that the publishable artifacts are complete and methodologically compatible.

## Recommended runtime

Use an NVIDIA GPU runtime. A 16 GB T4 is the practical baseline; an L4 or A100 gives more headroom. The notebook detects GPU memory and defaults activation collection to batch size 8 below 20 GB VRAM and 16 otherwise.

Runtime varies with Colab allocation, downloads, and prompt lengths. For planning, budget roughly **1–2 hours on a T4** for a first complete run, including model/SAE downloads; faster GPUs can be substantially quicker. Treat this as a planning estimate, not a benchmark.

## Persistence model

Colab VMs are temporary. The notebook mounts Google Drive and replaces the repo's `artifacts/` directory with a symlink to a Drive-backed directory. This means:

- activation artifacts survive a runtime reset;
- task-level causal and feature-set checkpoints survive a reset;
- `python -m experiments.run_all --resume` can continue rather than restart completed work;
- Hugging Face model caches remain on the Colab VM for speed and may need to be downloaded again after a new runtime.

The causal and feature-set stages checkpoint after every completed task in v0.15. Completion-marker files are runtime bookkeeping and are excluded from the publishable bundle.

## Notebook configuration

At the top of the notebook set:

```python
REPO_URL = "PASTE_YOUR_GIT_REPO_URL_HERE"
BRANCH = "main"
DRIVE_RUN_NAME = "FeatureLens_offline_v015"
```

`REPO_URL` can be the public Git URL of the Hugging Face Space repository or another Git mirror containing the same FeatureLens source.

If your default branch is not `main`, change `BRANCH`.

## If activation collection runs out of memory

The notebook chooses a conservative batch size automatically. If CUDA still runs out of memory, rerun the pipeline cell with:

```bash
python -m experiments.run_all --resume --activation-batch-size 4
```

Changing the activation batch size changes memory/time trade-offs, not the experiment definition.

## After the run

The notebook runs:

```bash
python -m scripts.validate_artifacts
```

and creates a small archive containing only the publishable study outputs. It excludes:

- `artifacts/activations/`;
- `.complete` checkpoint markers;
- model/SAE caches;
- temporary files.

Extract the archive over the local FeatureLens repository so the files land under `artifacts/`, then run the normal local release checks and push. The public **Study** tab will read those measured artifacts automatically.

## Publishable outputs

The bundle is expected to contain files such as:

- `artifacts/feature_catalog.csv`
- `artifacts/layer_metrics.csv`
- `artifacts/stability.csv`
- `artifacts/selection_stability.csv`
- `artifacts/causal_results_final_token.csv`
- `artifacts/causal_results_max_active.csv`
- `artifacts/causal_position_summary.csv`
- `artifacts/feature_set_results.csv`
- `artifacts/study_feature_summary.csv`
- `artifacts/study_summary.json`
- `artifacts/summary.json`
- `artifacts/report.md`
- `artifacts/figures/*.png`

Do not commit the `artifacts/activations/` directory.

## v0.16 causal addendum after a completed v0.15 study

If the full v0.15 Colab study already completed, **do not rerun the full notebook**. Use:

`notebooks/FeatureLens_Causal_Addendum_Colab.ipynb`

The addendum notebook copies only the small existing study outputs to a new Drive folder, preserves the final-token causal baseline, computes the 28-task `max_feature_activation` causal policy, regenerates the CPU study/report artifacts, validates them, and creates a new publishable ZIP.

The addendum does not recollect 224-prompt activations, refit feature/probe evaluations, rerun stability resampling, or rerun the 1/3/5 feature-set stage.
