# FeatureLens

FeatureLens is a causal interpretability workbench for `Qwen/Qwen3-1.7B-Base` and **Qwen-Scope residual-stream sparse autoencoders**. It asks one question:

> **Do sparse features that predict a concept also causally influence model behaviour?**

The project separates representational evidence from causal evidence. A feature can classify a concept well without controlling the downstream continuation one might infer from that association.

## Measured study result

The committed offline study uses **224 discovery prompts** (112 paraphrase pairs across seven controlled concepts) and **28 causal tasks**.

- Selected SAE features averaged **0.962 held-out AUROC** (median **0.987**, bootstrap 95% CI **[0.927, 0.994]**).
- Dense final-token residual probes reached **1.000 macro AUROC** at layers 14 and 26.
- Paraphrases preserved weighted sparse representations much more strongly than exact sparse support: mean cosine **0.985** versus TopK Jaccard **0.326**.
- Selected features were active at the conventional final prompt token on only **28.6%** of causal tasks, but somewhere in the prompt on **82.1%**.
- At the final token, targeted SAE interventions were **1.52×** the norm-matched random-control effect on average, but task-level uncertainty included zero (paired advantage **+0.0026**, 95% CI **[-0.0005, +0.0063]**, sign-flip **p=0.1719**).
- When intervention positions were chosen only from the selected feature's **maximum SAE activation within the prompt**, coverage rose to **82.1%** and targeted effects averaged **2.33×** matched-random controls (paired advantage **+0.0237**, 95% CI **[+0.0067, +0.0469]**, sign-flip **p≈1×10⁻⁴**).
- Final-token top-5 joint ablation produced only a **1.09×** SAE/random ratio, so adding more associated features did not automatically yield stronger causal specificity.
- Across the seven concepts, held-out AUROC and max-active target specificity had only weak descriptive association (**Spearman ρ=-0.185**).

The main conclusion is therefore not that predictive SAE features are automatically causal. **Causal evidence depended strongly on where the representation was tested**, and predictive strength by itself was a poor proxy for random-normalized causal specificity across concepts.

See [`artifacts/report.md`](artifacts/report.md) and the **Study** tab for the full measured result.

## Live workbench

FeatureLens loads Qwen-Scope SAEs for residual layers **4, 14, and 26** and supports:

- token-level residual capture and TopK feature inspection;
- SAE reconstruction diagnostics and layer trajectories;
- single-feature ablation, scaling, and decoder-direction injection;
- exact full-continuation teacher-forced scoring;
- next-token distribution shifts and deterministic generation comparison;
- eight norm-matched random controls for live specificity checks;
- scale dose-response and contrastive continuation preference;
- joint feature-set interventions, set-size sweeps, non-additivity, and decoder geometry;
- concept-guided candidate discovery with current-token causal readiness;
- batched candidate triage and controlled multi-candidate comparison;
- completion-cue, cue × context, token-trace, and controlled-concept diagnostics;
- local and prompt-wide paraphrase robustness;
- cross-target causal profiles and pairwise preference shifts.

Feature ids remain unlabeled unless there is empirical evidence for a concept association.

## Intervention semantics

For residual vector `h`, SAE coefficient `z_i`, decoder direction `d_i`, and multiplier `α`:

```text
ablate: h' = h - z_i d_i
scale:   h' = h + (α - 1) z_i d_i
inject:  h' = h + δ d_i
```

FeatureLens applies the decoded **delta** to the original residual instead of replacing the residual with the full SAE reconstruction. Batched causal experiments include a zero-edit condition in the same execution context, and random controls match the L2 norm of the targeted SAE perturbation.

## Offline study design

Concept evidence uses **prompt-wide maximum SAE activation across non-padding tokens**. Final-token sparse activations are saved separately for local analyses.

Causal evidence is reported under two position policies:

- **final token** — conventional final-prompt-token baseline;
- **max feature activation** — intervene where the selected feature is most strongly represented in that prompt.

Max-active positions are selected from SAE activation only, never from behavioral outcomes. Coverage is reported separately from effect strength.

The primary causal statistical unit is the **causal task**: ablation and 2× amplification are averaged within task before paired bootstrap and sign-flip inference.

The study additionally includes:

- train-only feature selection with held-out AUROC/F1;
- dense final-token residual linear-probe baselines;
- paraphrase robustness;
- 128-resample candidate-selection sensitivity;
- top-1/3/5 feature-set causal diagnostics;
- norm-matched random residual controls;
- cross-concept association-versus-causality synthesis.

## Reproducing the study

The canonical command is:

```bash
python -m experiments.run_all --resume
```

On a 16 GB GPU, a smaller activation batch is usually more comfortable:

```bash
python -m experiments.run_all --resume --activation-batch-size 8
```

After the expensive model stages exist, regenerate only CPU analysis with:

```bash
python -m experiments.run_analysis_only
```

Validate the final artifact bundle with:

```bash
python -m scripts.validate_artifacts
```

### Google Colab

Use [`notebooks/FeatureLens_Offline_Study_Colab.ipynb`](notebooks/FeatureLens_Offline_Study_Colab.ipynb) for a fresh full reproduction.

[`notebooks/FeatureLens_Causal_Addendum_Colab.ipynb`](notebooks/FeatureLens_Causal_Addendum_Colab.ipynb) is retained as the exact migration path used to extend an already-completed final-token baseline with the max-active causal study without recollecting discovery activations.

See [`notebooks/README.md`](notebooks/README.md) and [`docs/COLAB.md`](docs/COLAB.md).

## Public artifacts

The repository commits only small measured outputs. Large activation caches and model/SAE weights are excluded.

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
├── split.json
└── figures/
```

The **Study** tab reads these artifacts directly and does not rerun the model.

## Repository layout

```text
FeatureLens/
├── app.py
├── featurelens/          # SAE/runtime/intervention/study code
├── experiments/          # offline collection, causal study, analysis, reports
├── data/                 # controlled discovery prompts and causal tasks
├── artifacts/            # committed measured study outputs
├── notebooks/            # Colab study runners
├── scripts/              # validation and UI smoke checks
├── tests/
├── docs/
├── DESIGN.md
└── research_config.json
```

## Validation

```bash
python3 -m pytest -q
python3 -m compileall -q app.py featurelens experiments scripts
python3 -m ruff check app.py featurelens experiments tests scripts
python3 scripts/ui_smoke.py
python3 scripts/release_check.py
python3 -m scripts.validate_artifacts
```

The UI smoke test performs a real local Gradio `launch()`.

## Interpretation guardrails

- Held-out AUROC/F1 measure **concept association**, not causal influence.
- Max-active intervention positions are selected from SAE activation only, never from behavioral effect size.
- Cross-concept correlations use only seven concepts and are descriptive.
- Per-concept causal task counts are small; the primary inference pools task-level paired effects across all 28 causal tasks.
- Five causal tasks contained no activation of the selected feature anywhere in the prompt; max-active coverage was therefore 82.1%, not 100%.
- Dense linear probes and prompt-wide SAE features use different pooling schemes and are separate baselines.
- Prompt-wide max pooling discards token order.
- Live eight-control empirical tails are coarse diagnostics; the offline study is the primary aggregate evidence.

## Design

The public UI follows [`DESIGN.md`](DESIGN.md): restrained typography and color, flat information hierarchy, compact actions, explicit result headings, and minimal decorative chrome.

## License

MIT. See [`LICENSE`](LICENSE).
