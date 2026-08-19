---
title: FeatureLens
emoji: 🔬
colorFrom: gray
colorTo: green
sdk: gradio
python_version: "3.12.12"
sdk_version: "6.24.0"
app_file: app.py
pinned: false
license: mit
---

# FeatureLens

FeatureLens is a causal interpretability workbench for `Qwen/Qwen3-1.7B-Base` and the **Qwen-Scope residual-stream sparse autoencoders**. It is built around one question:

> **Do sparse features that predict a concept also causally influence model behaviour?**

The project keeps association and intervention evidence separate. A large SAE activation or strong held-out classifier is useful evidence about representation; a causal claim requires changing the residual stream and measuring downstream behaviour against controlled perturbations.

## What the live app does

FeatureLens loads SAEs for residual layers **4, 14, and 26** and supports:

- token-level residual capture and TopK SAE feature inspection;
- reconstruction cosine, NMSE, activation mass, and layer trajectories;
- single-feature ablation, scaling, and decoder-direction injection;
- exact full-continuation teacher-forced scoring;
- next-token distribution shifts and deterministic generation comparison;
- eight norm-matched random controls for live specificity checks;
- scale dose-response curves;
- contrastive continuation preference;
- joint feature-set interventions, 1/3/5 set-size sweeps, non-additivity, and decoder geometry;
- concept-guided candidate discovery with current-token causal readiness;
- batched candidate triage and random-controlled candidate comparison;
- token traces, completion-cue tests, cue × context tests, and controlled concept contrasts;
- local and prompt-wide paraphrase robustness;
- cross-target causal profiles and pairwise preference shifts.

The interface is deliberately an analytical tool rather than an SAE label viewer. Feature ids remain unlabeled until there is empirical evidence for a concept association.

## Intervention semantics

For residual vector `h`, SAE coefficient `z_i`, decoder direction `d_i`, and multiplier `α`:

```text
ablate: h' = h - z_i d_i
scale:   h' = h + (α - 1) z_i d_i
inject:  h' = h + δ d_i
```

FeatureLens applies the decoded **delta** to the original residual. It does not replace the residual with the full SAE reconstruction, so reconstruction error is not silently mixed into the intervention.

Batched causal experiments include a zero-edit condition in the same execution context. Random controls match the L2 norm of the SAE perturbation.

## Offline study

The live app is exploratory. The offline study is the dataset-scale experiment.

It uses **224 discovery prompts** arranged as 112 paraphrase pairs across seven controlled concepts, plus **28 separate causal tasks**. Concept evidence uses prompt-wide max-pooled SAE activations across non-padding tokens; final-token sparse activations are saved separately for local analyses.

Causal evidence is reported under two position policies: the original **final-token** baseline and **max-feature-activation**, which patches the selected feature where it is most strongly represented in the prompt. Max-active positions are selected from SAE activation only, never from downstream behavioral effects. Primary uncertainty uses the causal task as the statistical unit.

The study produces:

- train-only feature selection with held-out AUROC/F1;
- a dense final-token residual linear-probe baseline;
- paraphrase stability;
- 128-resample candidate-selection sensitivity;
- random-controlled single-feature causal results under both final-token and max-feature-activation patch policies;
- top-1/3/5 feature-set causal results;
- causal-position coverage/sensitivity and cross-concept association-versus-causality synthesis;
- uncertainty-aware report figures and a measured Markdown report.

Run the full pipeline with:

```bash
python -m experiments.run_all --resume
```

On a memory-constrained GPU, activation collection can be tuned without changing the experiment definition:

```bash
python -m experiments.run_all --resume --activation-batch-size 8
```

If you already completed the v0.15 study, upgrade it with only the positional causal addendum:

```bash
python -m experiments.run_causal_addendum --resume
```

After the expensive model stages exist, CPU-only analysis can be regenerated with:

```bash
python -m experiments.run_analysis_only
```

Artifact integrity is checked with:

```bash
python -m scripts.validate_artifacts
```

### Google Colab

A ready-to-run full-study notebook is included at [`notebooks/FeatureLens_Offline_Study_Colab.ipynb`](notebooks/FeatureLens_Offline_Study_Colab.ipynb). If the v0.15 study is already complete, use [`notebooks/FeatureLens_Causal_Addendum_Colab.ipynb`](notebooks/FeatureLens_Causal_Addendum_Colab.ipynb) instead; it runs only the max-active causal addendum and CPU synthesis.

See [`docs/COLAB.md`](docs/COLAB.md) for the exact workflow.

## Public artifacts

Large activation caches are intentionally excluded from Git. The small measured outputs that can be committed after a real run include:

```text
artifacts/
├── feature_catalog.csv
├── layer_metrics.csv
├── stability.csv
├── selection_stability.csv
├── causal_results_final_token.csv
├── causal_results_max_active.csv
├── causal_position_summary.csv
├── feature_set_results.csv
├── study_feature_summary.csv
├── study_summary.json
├── summary.json
├── report.md
└── figures/
```

The **Study** tab reads these artifacts directly. Before the offline run is materialized it intentionally shows no placeholder metrics.

## Repository layout

```text
FeatureLens/
├── app.py                    # Gradio / ZeroGPU workbench
├── featurelens/              # SAE, runtime, metrics, interventions, study loader
├── experiments/              # offline collection, evaluation, causal study, reports
├── data/                     # controlled prompt and causal-task definitions
├── artifacts/                # small public study outputs; activations are ignored
├── notebooks/                # Colab runner
├── scripts/                  # release, UI smoke, artifact validation
├── tests/                    # software and methodology regression tests
├── docs/                     # methodology, validation, deployment, Colab notes
├── DESIGN.md                 # UI design contract
└── research_config.json      # experiment configuration
```

## Local validation

```bash
python3 -m pytest -q
python3 -m compileall -q app.py featurelens experiments scripts
python3 -m ruff check app.py featurelens experiments tests scripts
python3 scripts/ui_smoke.py
python3 scripts/release_check.py
```

The UI smoke test performs a real local Gradio `launch()` rather than only constructing the component tree.

## Methodology notes

Several quantities answer different questions and should not be collapsed into one score:

- **held-out AUROC/F1** — concept association;
- **paraphrase / resample stability** — sensitivity to wording or sample choice;
- **target Δ log p** — effect on one specified continuation;
- **Jensen-Shannon divergence** — local distributional change;
- **random-normalized specificity** — whether the targeted SAE edit exceeds an equal-norm residual perturbation baseline;
- **feature-set non-additivity** — downstream interaction under joint edits;
- **decoder geometry** — alignment/cancellation before downstream model non-linearity.

The full methodology is documented in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) and the offline study protocol in [`docs/OFFLINE_STUDY.md`](docs/OFFLINE_STUDY.md).

## Limitations

- Live ZeroGPU controls are intentionally small; eight-control empirical tails are coarse diagnostics.
- SAE feature ids are layer-specific and should not be compared across layers by id.
- Prompt-wide max pooling discards token order.
- Dense linear probes and prompt-wide SAE features use different pooling schemes and are reported as separate baselines.
- Cross-concept study correlations have only seven concepts and are descriptive.
- A candidate feature can be predictive without being causally specific, and a causally disruptive feature need not selectively control the target one might infer from its association.

## Design

The public UI follows the project-specific design contract in [`DESIGN.md`](DESIGN.md): restrained typography and color, flat information hierarchy, minimal decorative chrome, compact actions, explicit table headings, and no marketing-style cards/badges/gradients.

## License

MIT. See [`LICENSE`](LICENSE).
