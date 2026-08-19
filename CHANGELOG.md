# Changelog

## v0.15.0

- Reworked the public Gradio surface around a documented **research-instrument design system** rather than SaaS/dashboard defaults.
- Added `DESIGN.md` with typography, color, spacing, surface, button, table, plot, and anti-pattern rules so future UI edits have explicit constraints.
- Replaced the three-column onboarding/card pattern with compact editorial guidance; flattened repeated context/callout treatment; removed visible release marketing; shortened result copy so measured values lead and methodology lives in the Method tab/docs.
- Introduced a two-typeface hierarchy (serif display headings, neutral sans-serif controls/data), tighter semantic spacing, compact primary actions, quiet utility buttons, and stronger explicit table headings.
- Normalized the dynamic cross-target plots to a restrained three-series palette instead of Vega's saturated categorical defaults.
- Rewrote the public README around the research question, live tool, offline study, and reproducible workflow instead of a long release-history narrative.
- Added a ready-to-run Google Colab notebook plus `docs/COLAB.md` for Drive-backed artifact persistence and resumable study execution.
- Added `--activation-batch-size` / `--activation-max-length` to the full runner and task-level checkpoint/resume support inside causal and feature-set stages.
- Added automated design-contract regression tests.

## v0.14.0

- Transitioned the project from live-feature expansion toward the full offline empirical study.
- Changed offline SAE concept evidence from final-token-only activations to **prompt-wide max-pooled activations across non-padding tokens**, while saving separate final-token sparse activation matrices for local diagnostics.
- Added `experiments/analyze_stability.py` with 128 deterministic balanced activation resamples and per-feature shortlist support/rank summaries.
- Added `experiments/analyze_study.py` to join held-out AUROC/F1, paraphrase robustness, candidate stability, feature activity, and random-normalized target/JS causal specificity by controlled concept.
- Added descriptive cross-concept association-vs-causality correlations and new association/candidate-stability report figures.
- Expanded the public **Offline study** tab into an artifact-backed results dashboard that remains explicitly empty until real study artifacts are committed.
- Added `python experiments/run_all.py --resume` for interrupted/preemptible GPU sessions and `python experiments/run_analysis_only.py` for CPU-only re-analysis once inference artifacts exist.
- Added `scripts/validate_artifacts.py` to verify prompt-wide activation provenance and public study artifact schemas before commit.
- Added dedicated offline-study methodology/validation documentation and automated tests for prompt-wide pooling, stability scoring, task-paired specificity, study UI states, and correlation guardrails.

## v0.13.0

- Added deterministic **32-resample candidate-support diagnostics** from the same concept-discovery activation batch; each displayed feature now reports shortlist support and median resample rank without another model forward.
- Extended cross-target profiling with **normalized effect entropy, effect concentration, signed bias, and a descriptive profile pattern** so concentrated target dependence is separated from broad same-sign behavior.
- Added **pairwise target-preference shifts** derived from the same cross-target scores: Δ(A−B) = Δmean(A) − Δmean(B), with a table and plot and no additional inference.
- Kept HF acceptance quota-aware: only concept discovery and cross-target profiling are touched GPU paths.


## v0.12.0

- Added **split-half discovery stability** using the already-computed concept activation batch, so shortlist sensitivity is visible without another GPU forward.
- Added **Controlled evidence patterns**, a zero-GPU synthesis that distinguishes broad controlled influence, target-weighted effects, distribution-shift-dominant effects, and weak/mixed specificity while keeping eight-control tails explicitly coarse.
- Changed **Association vs controlled causality** so missing discovery state after a Space rebuild produces an explicit explanation instead of a blank panel.
- Added **Cross-target causal profile** for up to three candidates and five exact continuations, screening whether native ablation effects concentrate on one target or generalize across alternatives.
- Added automatic cross-target shortlist handoff from the target-specificity and JS-specificity leaders.
- Kept the validated in-place focus behavior unchanged.
- Kept HF validation quota-aware: only the touched discovery path and the new cross-target path need live GPU acceptance.

## v0.11.0

- Added **Controlled candidate specificity**, a one-batch follow-up that compares up to three candidate SAE ablations against each candidate's own 8-direction norm-matched random ensemble.
- Added a strategic controlled shortlist that preserves the discovery leader, target-effect leader, and distribution-shift leader when they differ, then fills remaining slots by triage target rank.
- Added target-specificity and JS-specificity ratios plus coarse empirical random-control tails for every controlled candidate.
- Added **Association vs controlled causality**, joining concept-discovery evidence to random-normalized causal specificity instead of relying only on raw triage magnitude.
- Kept target-specific and whole-distribution causal influence separate rather than collapsing them into one score.
- Kept the validated in-place focus/zoom implementation unchanged.
- Reduced HF acceptance to one new GPU call; unchanged discovery/triage and other regression paths remain covered by automated tests.

## v0.10.0

- Added a zero-extra-GPU **Discovery–causality alignment** panel after candidate triage.
- Joined discovery rank/score with target-effect rank and next-token-distribution-shift rank for the same screened candidates.
- Added descriptive Spearman ρ summaries for candidate evidence vs absolute target effect and vs next-token JS.
- Added an association-evidence-vs-target-effect scatter plot and rank-shift table.
- Kept the v0.9 in-place focus implementation unchanged after HF validation.
- HF validation remains GPU-budget-aware: only the discovery and candidate-triage paths need to be exercised.

## v0.9.0

### In-place focus and layout polish
- Replaced the HF-iframe-hostile overlay/fullscreen experiment with **in-place focus** for plots and tables. The original component expands exactly where it is located; plots are scaled from their existing rendering so aspect ratio is preserved and no cloned toolbar icon can be mistaken for the chart.
- Focus is capped by both screen width and height, and the same fullscreen toolbar icon toggles the component back without moving the page to the top.
- Reworked explicit result-table headings into compact HTML headings that occupy the Dataframe toolbar whitespace instead of leaving a large empty band above the first row.
- Removed the unnecessary “Standalone experiment” dose-response explanation while keeping independent feature and target fields.

### Candidate-to-causality workflow
- Added **Batched causal candidate triage**. Up to eight concept-discovery candidates are independently ablated in one batched scoring run at the current Workbench location.
- The screen reports native activation, perturbation norm, target mean/sequence log-probability deltas, and next-token JS, and ranks candidates by absolute target effect.
- The triage deliberately omits random controls; its purpose is to identify which candidate is worth promoting to the existing single-feature 8-direction random-control test.
- Concept discovery now directly populates the candidate-screen multiselect, defaulting to up to five returned candidates.

### GPU-budget-aware validation
- HF acceptance no longer reruns identity paraphrase, layer trajectory, or 1/3/5 set-size sweeps when those code paths are unchanged. Automated tests cover them; scarce ZeroGPU minutes are reserved for new/touched inference paths.

## v0.8.0

### UI readability and focus
- Replaced plot-native fullscreen behavior with a **bounded FeatureLens focus overlay**. The plot is copied into a centered reading surface (max ~1120 px) instead of stretching across an ultrawide display; closing the overlay restores the original page position.
- Kept descriptive plot export filenames and the Gradio Dataframe fullscreen control.
- Replaced fragile native Dataframe labels with explicit **result-table headings** above every major table so table titles follow the same typography hierarchy as the rest of the application.
- Added a muted cue palette to the cue × context plot instead of relying on Gradio/Vega default saturated series colors.

### Independent experiment inputs
- The scale dose-response panel now owns its **Dose-response feature id** and **Dose-response target continuation**. It no longer depends on running the single-feature causal test or filling that section's optional target field first.
- Clarified in-panel provenance: dose response reads the current prompt/layer/token fields from Workbench Section I but is otherwise a standalone experiment.

### Candidate discovery
- Added **Causal-ready at current token** ranking. It requires positive concept contrast *and* activation at the selected Workbench token, then ranks those compatible candidates using balanced selectivity plus a log-scaled current-token activation term.
- Discovery summaries now report how many displayed candidates are actually active at the selected Workbench token. This makes the distinction between a prompt-wide concept candidate and an immediately ablatable feature explicit.
- Balanced selectivity and raw mean-difference modes remain available for methodological comparison.

### Cue specificity
- Cue × context summaries now derive the dominant cue, its context coverage, and off-dominant activity. A feature that fires for one cue in every tested context while all other cues stay inactive is reported as a **cue-dominant tested pattern**, not merely with generic interpretation text.

### Validation
- Expanded the automated suite to cover causal-ready candidate discovery, independent dose-response target state, cue-dominance diagnostics, plot-focus JavaScript markers, and explicit result-heading behavior.
- Retained compile, Ruff, actual Gradio `launch()`, release-check, and deferred final adversarial-suite gates.

## v0.7.0

### Research-instrument UI cleanup
- Removed collapsible wrappers from the core **scale dose-response** and **contrastive preference** experiments so section headings are no longer duplicated by accordion titles.
- Simplified the page header and removed visible footer/redundancy that did not help a reviewer use the tool.
- Strengthened result-table title and column-header typography.
- Replaced full-viewport plot stretching with a bounded top-centered **focus view**; exiting focus restores the prior page position.
- Kept native plot export but rename downloads to descriptive `featurelens_<plot-name>.png` filenames instead of a generic chart name.

### Candidate discovery and causal readiness
- Reworked live concept-guided discovery around **Balanced selectivity**: `selectivity × target activation rate × log1p(target mean)`. This prevents very large but non-selective SAE coefficients from dominating the exploratory shortlist.
- Retained **Raw mean difference** as an explicit comparison mode rather than silently changing the old ranking.
- Candidate discovery now evaluates the current Workbench prompt in the same GPU batch and reports current-prompt maximum activation plus selected-token activation.
- The candidate selector defaults to the highest-ranked displayed candidate active at the current Workbench token when one exists.
- Candidate-table row selection is wired directly to the candidate selector, and the reuse action now confirms exactly which downstream feature selectors were updated.

### Lexical / structural specificity
- Added a **cue × context specificity matrix**: cross several prompt stems with the same completion cues in one batched forward and measure the selected feature at every resulting final token.
- This extends the single-stem completion-cue test so a response to `is` can be separated from a broader completion-boundary or context-dependent response.

### Controlled data and validation
- Replaced the French-language control concept one-for-one with a **German-language** concept while preserving 224 balanced discovery prompts and 28 causal tasks.
- Added regression coverage for German data, balanced/raw candidate ranking, current-Workbench candidate compatibility, cue × context scanning, candidate row selection, descriptive export naming, and focus-position preservation.
- Kept the actual Gradio `launch()` smoke test, compile gate, release checker, and deferred comprehensive adversarial suite.

## v0.6.0

### UX / navigation
- Added a plain-language **Start here** tab with a three-step workflow and glossary for non-specialist reviewers.
- Added a persistent **Current Workbench context** banner so inherited prompt/layer/token state is visible from every tab.
- Added explicit editable feature selectors for **scale dose-response** and **contrastive continuation preference** instead of silently reusing the single-feature selector.
- Clarified state provenance in Feature Sets and Feature Evidence; experiment text now states whether it inherits Workbench context or uses an independent prompt set.
- Normalized heading hierarchy and increased table/header typography for readability.
- Added native plot **fullscreen** and **export PNG** controls to every BarPlot/LinePlot.

### Live research tools
- Added **concept-guided candidate feature discovery**: rank features for a selected controlled concept using prompt-wide target-minus-other mean maximum activation, with selectivity and activation-rate diagnostics.
- Added a **completion-cue sensitivity scan** that appends controlled suffixes to a prompt stem and measures the selected feature at the resulting final token.
- Added a one-click action to reuse a discovered candidate across single-feature, dose-response, contrastive, and evidence feature selectors.
- Candidate discovery and cue scans are explicitly exploratory; neither creates semantic labels or overwrites `Offline concept hint`.

### Validation
- Expanded toy-runtime coverage to **40 tests**, including candidate-feature discovery and completion-cue scans.
- Retained the actual Gradio `launch()` smoke gate, compile gate, release checker, batched-null regression tests, and final-release adversarial-test deferral.

## v0.5.0

### Causal specificity
- Added a **contrastive continuation preference test** that scores two exact continuations under the same single-feature intervention and 8-direction norm-matched control ensemble.
- Reports baseline/edited sequence log-odds A−B, causal log-odds shift, token-normalized preference shift, random-control magnitude statistics, and an exploratory empirical tail probability.
- Keeps this distinct from absolute target probability so broad distributional disruption is not mistaken for selective behavioral control.

### Feature evidence and geometry
- Added a **feature-token activation trace** over every token in the current Workbench prompt.
- Fixed the controlled concept scan to use **prompt-wide max activation over non-padding tokens** instead of only the final token.
- All-zero concept batches now report `inactive in every sampled prompt` and do not invent a leading concept.
- Added **feature-set decoder geometry** for 2–8 selected features: pairwise decoder cosine, mean/max absolute cosine, activation-weighted joint-ablation norm, independent-direction reference norm, and alignment/cancellation ratio.

### Interface
- Widened and explicitly centered the application canvas (up to 1600 px) and enabled `fill_width=True` to use desktop space more effectively.
- Normalized serif typography, labels, controls, table font sizes, and action-button styling.
- Added bounded Dataframe heights to reduce excessive dynamic page growth.
- Added copy-button visual acknowledgement (`✓ Copied with headers`).
- Added a browser-side resize/mutation observer to request layout reflow when dynamic output height changes inside an embedded Space.
- Retained the safe Gradio theme configuration without string font tuples; serif typography is applied in CSS.

### Validation
- Expanded automated coverage from 29 to **38 tests**, including contrastive log-odds, decoder geometry, copy/export helpers, and toy-runtime end-to-end checks.
- Added `scripts/ui_smoke.py` so the actual Gradio `launch()` path is part of the release procedure instead of only constructing the component tree.
- Reworked v0.5 acceptance tests around the new concept-scan semantics, feature-token trace, contrastive preference, geometry, copy feedback, and embedded-page reflow.

## v0.4.0

### Causal correctness
- Added an explicit **batched zero-edit reference** to live causal batches so intervention effects are measured against the same execution context as edited rows.
- Changed the scale dose-response reference to the batched `1×` row. The `1×` row is therefore an exact causal no-op by construction rather than a separately executed baseline comparison.
- Added execution-context drift diagnostics so any remaining single-forward vs batched-forward numerical difference is reported as instrumentation drift, not causal signal.

### Stronger negative controls
- Replaced the single live random residual direction with an **8-direction norm-matched random ensemble**.
- Single-feature, joint feature-set, and 1/3/5 set-size experiments now report random signed mean, mean absolute effect, standard deviation, targeted/random magnitude ratio, and a small-sample empirical tail probability.
- Updated offline causal and feature-set runners to use the same zero-edit reference and configurable random-control ensembles.
- Updated report pairing so each targeted intervention is compared with the mean absolute effect of its complete random-control ensemble rather than an arbitrary first control.

### Distributed causality
- Added **individual-vs-joint ablation decomposition** for 2–5 features.
- Reports the individual effects, additive expectation, observed joint effect, interaction excess, and normalized non-additivity.
- The UI explicitly treats non-additivity as a diagnostic, not proof of a direct feature-feature circuit.

### Representation robustness and evidence
- Added **prompt-wide paraphrase robustness** using max activation per SAE feature across all prompt tokens, alongside the stricter selected-token comparison.
- Added a live **controlled concept contrast scan** over the seven balanced discovery concepts for one selected feature.
- Concept contrast results remain exploratory and never overwrite `Offline concept hint` or claim a semantic label from the live scan.

### UI and export
- Reworked the visual language toward a restrained, print-inspired interface with serif typography, flatter controls, thin rules, and muted chart colors.
- Replaced intervention radio pills with conventional dropdown controls and realigned the Feature Sets form.
- Added extra bottom spacing plus an explicit end-of-workbench footer to avoid an app-controlled abrupt cutoff in embedded Spaces.
- Added **Copy table with headers** actions for every major output table; copied text is tab-separated and begins with the column names.
- Added a dedicated **Feature evidence** tab for controlled feature-concept contrast tests.

### Validation
- Expanded the software suite to **29 tests**.
- Added report tests that verify random-control ensembles are aggregated correctly before paired causal statistics are computed.
- Rewrote `docs/VALIDATION.md` around exact v0.4 UI labels, including explicit numerical-null, table-copy, adversarial, responsive-layout, and queue tests.

## v0.3.0

### Causal measurement
- Replaced first-token-only target evaluation with exact full-continuation teacher-forced log-probability scoring.
- Added total sequence and mean-per-token log-probability deltas plus per-target-token decomposition.
- Retained next-token probability/JS diagnostics and greedy generation as complementary outputs.

### Distributed feature causality
- Added joint multi-feature ablation/scaling using summed reconstruction-preserving SAE decoder deltas.
- Added live top-1/top-3/top-5 joint-ablation sweep with norm-matched random controls.
- Added offline `experiments/run_feature_sets.py` and report integration.

### Robustness
- Added a live paraphrase-robustness explorer with TopK Jaccard, sparse cosine, overlap table, and activation comparison.

### Efficiency
- Batched all six single-feature dose-response edits into one model forward after the baseline.
- Batched targeted/random 1/3/5 feature-set sweep conditions into one model forward after the baseline.

### UI / deployment
- Replaced the bright blue visual emphasis with muted teal/stone accents and explicit chart palettes.
- Added visible `Prompt tokens` headings and aligned validation terminology with actual UI labels.
- Explicitly labels the dose-response panel as a scale intervention: 0× = ablation, 1× = no edit.
- Kept SSR disabled for the Hugging Face Space.

## v0.2.0
- Added live norm-matched random controls.
- Added single-feature causal dose-response.
- Added layer trajectory diagnostics.
- Added bootstrap confidence intervals and paired sign-flip tests.
- Hardened Gradio / ZeroGPU deployment.
