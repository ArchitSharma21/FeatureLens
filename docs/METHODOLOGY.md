# FeatureLens methodology

## Primary question

FeatureLens separates **representation** from **causal influence**.

A sparse feature can predict a concept because it correlates with useful information in the residual stream. That does not imply that changing that feature direction will specifically change model behaviour. The project therefore evaluates four distinct properties:

1. reconstruction quality;
2. predictive association;
3. robustness across paraphrases;
4. causal sensitivity under controlled residual interventions.

---

## Model and sparse dictionaries

The live workbench uses `Qwen/Qwen3-1.7B-Base` and Qwen-Scope residual-stream SAEs at layers `4`, `14`, and `26`.

Each selected residual location is encoded with the corresponding layer-specific SAE. Feature IDs are meaningful only **within one SAE dictionary**; numerical IDs must not be compared as if they were shared semantics across layers.

---

## Discovery data

The controlled discovery benchmark contains seven concept groups:

- code;
- mathematics;
- positive sentiment;
- negative sentiment;
- German language;
- factual entities;
- uncertainty.

Each concept contains 16 paraphrase pairs, two prompts per pair, for 224 prompts total.

Train/test splitting is grouped by `pair_id`, so the two paraphrases from the same source pair cannot leak across train and held-out test.

The offline discovery benchmark uses the final prompt-token residual. The live workbench remains token-selectable.

---

## Sparse feature discovery

For each configured layer, FeatureLens records the TopK SAE code at the chosen prompt location.

Candidate features are selected **using the training split only**. Training AUROC is the primary selection score, with activation-rate contrast used as a tie-break. Held-out AUROC and F1 are computed only after feature selection.

This prevents the project from selecting whichever SAE feature happened to look best on held-out data.

---

## Dense linear-probe baseline

A multinomial logistic-regression probe is fit to dense residual vectors from the same layers.

The comparison asks whether concept information is linearly available in the dense representation even when no individual sparse feature cleanly isolates it.

A result such as strong linear-probe performance but weak single-feature AUROC would support a more distributed representation rather than a clean monosemantic feature story.

---

## Reconstruction diagnostics

For residual `h` and SAE reconstruction `h_hat`, FeatureLens records:

- reconstruction cosine;
- normalized mean squared error;
- active feature count;
- Top-5 activation mass;
- normalized sparse activation entropy in the live layer trajectory.

These are diagnostics, not causal evidence. Reconstruction quality can also vary substantially by layer.

---

## Reconstruction-preserving residual interventions

For selected SAE activation `z_i`, decoder direction `d_i`, and scale multiplier `α`:

```text
ablate:   h' = h - z_i d_i
scale:    h' = h + (α - 1) z_i d_i
inject:   h' = h + δ d_i
```

Only the decoded **feature delta** is added to the original residual stream. FeatureLens does **not** replace the residual with the complete SAE reconstruction.

This avoids conflating the intervention with full-reconstruction error.

`inject` is intentionally interpreted differently from ablation/scaling: it tests an externally supplied decoder-direction steering coefficient even if that feature was not naturally active.

---

## Full-continuation teacher-forced scoring

A target such as `2x` may consist of multiple tokenizer tokens. FeatureLens therefore scores the complete target continuation, not only its first token.

The exact target token IDs are appended to the prompt. Teacher-forced model logits provide a log probability for every target token.

The live and offline causal metrics include:

- per-target-token log probability;
- total target sequence log probability;
- mean target log probability per token;
- intervention-minus-reference deltas.

The primary aggregate causal metric is **Δ mean target log probability per token**, because it is less directly dependent on target length than total sequence log probability.

Next-token Jensen-Shannon divergence and greedy generation remain complementary diagnostics. Identical greedy outputs do not imply an intervention had zero probability-level effect.

---

## Batched zero-edit causal reference

v0.3 exposed an important numerical issue: batched and separately executed model forwards can differ slightly even when the intended residual edit is zero. A nominal `1×` dose-response condition therefore showed a small non-zero effect when compared with a separately executed baseline.

FeatureLens treats this as instrumentation drift, not causal evidence.

Every batched causal experiment now includes an explicit zero-edit condition:

```text
Δh = 0
```

All intervention deltas in that batch are measured against this **same-execution-context reference**.

For the scale dose-response, the `1×` row is the zero-edit reference itself, so its causal deltas are exactly zero by construction.

When a separately executed capture forward is also needed, FeatureLens reports the single-vs-batch discrepancy as **execution-context null drift**. That diagnostic is kept separate from causal effect size.

---

## Norm-matched random-control ensemble

A single random residual direction is a fragile negative control: by chance it can be unusually weak or unusually disruptive.

For a targeted perturbation `Δh_sae`, FeatureLens generates deterministic random directions `r_j` and rescales each one so that:

```text
||r_j||_2 = ||Δh_sae||_2
```

The live app uses **8 random directions** for each targeted perturbation. Targeted and control conditions are evaluated in the same batched execution context.

The live app reports:

- signed random mean effect;
- mean absolute random effect;
- random-effect standard deviation;
- `|targeted effect| / mean(|random effect|)`;
- an empirical two-sided magnitude tail quantity:

```text
(1 + count(|random_j| >= |targeted|)) / (N + 1)
```

With only eight live controls this empirical value is intentionally coarse: the smallest possible value is `1/9`. It is an exploratory specificity diagnostic, **not** a conventional significance test.

Offline runs can increase `--random-controls` when compute allows.

---

## Single-feature scale dose-response

The live dose-response experiment always evaluates:

```text
0×, 0.5×, 1×, 1.5×, 2×, 3×
```

where:

- `0×` = complete ablation;
- `1×` = zero edit;
- `2×` = double the native coefficient.

All six edited residual conditions are stacked along the batch dimension and scored together. The `1×` batch row is the reference for every reported delta.

A monotonic relationship would strengthen a simple directional causal interpretation, but monotonicity is not required. Non-monotonic responses are retained as evidence about the model rather than “corrected” away.

---

## Joint feature-set interventions

A concept may be distributed across multiple sparse features.

For a same-layer feature set `S`, FeatureLens sums the individual decoder-direction deltas:

```text
h' = h + Σ_i∈S Δz_i d_i
```

The live app supports joint ablation and shared-multiplier scaling. Cross-layer feature sets are never combined into one residual intervention.

Additive injection is deliberately omitted from the feature-set UI because one shared additive coefficient across unrelated decoder directions has no uniquely natural interpretation.

---

## 1 / 3 / 5 feature-set sensitivity

At the selected prompt location, FeatureLens jointly ablates the strongest active:

```text
k = 1, 3, 5
```

features.

Each targeted joint perturbation receives its **own 8-direction norm-matched random ensemble**. The targeted edit, random controls, and zero-edit reference are scored in a batched execution context.

Increasing `k` is not assumed to increase effect magnitude or specificity. Non-monotonic behaviour may indicate redundancy, cancellation, distributed representation, or ordinary model non-linearity.

---

## Individual-vs-joint non-additivity

For a selected set of 2–5 active features, FeatureLens separately evaluates:

```text
feature 1 ablation
feature 2 ablation
...
joint ablation
zero-edit reference
```

Let `e_i` be the Δ mean target log probability/token for the individual ablation of feature `i`, and `e_joint` the joint effect.

FeatureLens reports:

```text
additive expectation = Σ_i e_i
interaction excess   = e_joint - Σ_i e_i
```

and a normalized interaction value scaled by the sum of absolute individual effects.

A non-zero excess diagnoses **non-additivity under this intervention**. It does not establish that the selected SAE features directly interact with one another or form a mechanistic circuit; downstream nonlinearities can also produce non-additivity.

---

## Paraphrase robustness: selected-token and prompt-wide

The strict selected-token comparison measures:

- TopK feature-support Jaccard;
- sparse activation cosine.

This can be misleading when the two manually selected tokens play different roles. For example, comparing the final token `is` in one prompt with a final punctuation token in its paraphrase is not a clean semantic-anchor comparison.

FeatureLens therefore also reports a prompt-wide profile. For each SAE feature, FeatureLens takes its maximum activation across all prompt tokens:

```text
profile_i(prompt) = max_t z_{t,i}
```

The two prompt-wide sparse dictionaries are then compared by support Jaccard and cosine.

The selected-token and prompt-wide metrics answer different questions and are intentionally displayed together.

---

## Feature-token activation trace

For a selected SAE feature and layer, FeatureLens can encode every prompt-token residual and report the feature's TopK activation token by token. This answers a basic localization question that a single selected-token view cannot: is the feature concentrated at one syntactic/semantic position, or does it recur across the prompt?

The trace reports activation at each token, active-token count, and the peak token. A zero entry means that feature is not present in that token's TopK SAE support.

---

## Prompt-wide controlled concept contrast scan

The live **Feature evidence** tab also provides an exploratory concept contrast for one selected feature.

It samples a small balanced batch from the same seven controlled concept groups, using only one wording from each paraphrase pair. Earlier versions sampled only each prompt's final token, which can miss a feature that is active elsewhere in the prompt. the current live scan instead defines each prompt-level feature score as:

```text
score(feature, prompt) = max over non-padding prompt tokens of z_feature
```

For each concept it reports:

- mean prompt-wide maximum activation;
- median prompt-wide maximum activation;
- prompt activation rate;
- mean activation when active;
- maximum activation.

If the feature is inactive in every sampled prompt, FeatureLens reports that state explicitly and does not assign a leading concept.

This is a **diagnostic**, not the offline feature-labeling procedure. The live scan never modifies `Offline concept hint` and does not claim semantic identity from a handful of prompts.

---

## Contrastive continuation preference

Absolute probability change for one target can be difficult to interpret when an SAE edit broadly perturbs the output distribution. v0.5 therefore adds a contrastive causal test for two exact continuations, A and B.

For each continuation, FeatureLens performs full teacher-forced sequence scoring under the same zero-edit reference, targeted SAE edit, and norm-matched random-control ensemble. It then computes:

```text
preference = log P(A) - log P(B)
causal preference shift = preference_edited - preference_reference
```

The exact-sequence quantity is a true log-odds comparison between those two specified continuations. The UI additionally reports a token-normalized preference difference for interpretability when the continuations have different lengths.

A large next-token JS effect but a weak contrastive preference shift is evidence for **general distributional influence** more than selective control of the A-vs-B behavioral choice.

---

## Feature-set decoder geometry

Joint interventions can be affected by the geometry of SAE decoder directions before any downstream non-linearity is considered. For 2–8 selected same-layer features, v0.5 reports pairwise decoder cosine similarities.

It also forms each feature's activation-weighted ablation delta and compares:

```text
||sum_i delta_i||_2
```

with the independent-direction reference:

```text
sqrt(sum_i ||delta_i||_2^2)
```

Their ratio is 1 for an orthogonal/independent norm geometry, below 1 under net cancellation, and above 1 under net alignment. This diagnostic helps interpret joint-ablation results but does not establish downstream causal interaction by itself.

---

## Statistical interpretation in the offline report

The generated offline report uses paired targeted-vs-control comparisons.

For a causal condition with several random controls, the report first aggregates the control ensemble for that same task/condition rather than selecting an arbitrary random row.

It then uses:

- bootstrap 95% confidence intervals;
- paired sign-flip randomization tests;
- targeted/random effect-size ratios.

A large point-estimate ratio alone is not sufficient for a strong causal-specificity narrative if paired uncertainty remains weak.

FeatureLens keeps the raw rows even when the resulting conclusion is null, mixed, or contrary to the original hypothesis.


## Live candidate discovery

The selected-feature concept contrast asks **given a feature, where does it activate?** FeatureLens also asks the reverse live question: **given a controlled concept, which SAE features are plausible candidates to investigate?**

For a balanced batch with `n` prompts from each of the seven controlled groups, FeatureLens encodes every non-padding token and forms a prompt-wide feature profile by taking the maximum TopK activation of each SAE feature over the prompt. For target concept `c` and feature `f`, it reports:

```text
target_mean(f) = mean prompt-wide max over prompts in c
other_mean(f)  = mean prompt-wide max over prompts outside c
mean_difference(f) = target_mean(f) - other_mean(f)
selectivity(f) = mean_difference(f) / (target_mean(f) + other_mean(f) + eps)
```

The default live ranking uses the exploratory score `max(0, selectivity) × target_activation_rate × log1p(target_mean)`, which prevents very large but broadly active SAE coefficients from dominating merely because of scale. A raw positive `mean_difference` ordering remains available for comparison. The table also reports target/other activation rates plus current-Workbench prompt-wide and selected-token activation. This is intentionally **not a held-out labeler**: the same small controlled batch is used for live screening. Semantic claims still require the offline grouped train/test feature evaluation.

## Completion-cue sensitivity

The v0.5 math example showed feature `22632` active only at the final `is` token. That pattern motivates a lexical/structural control. For one prompt stem and a user-supplied list of completion cues, FeatureLens appends each cue, encodes the resulting prompt, and measures the selected feature at the final non-padding token.

The cue scan is intended to distinguish hypotheses such as:

- concept-linked activation;
- lexical activation tied to a particular token such as `is`;
- structural activation at a completion boundary;
- broad activation across several continuation cues.

It is a controlled diagnostic only. A cue response cannot establish the feature's complete semantics.

## Cue × context specificity

A single completion-cue scan can show that a feature responds to a token such as `is`, but cannot distinguish a lexical cue feature from a context-dependent completion-boundary feature. v0.7 therefore crosses several prompt stems with the same cue set in one batched forward pass.

For feature $f$, stem $s$, and cue $c$, the diagnostic records the final-token SAE activation $z_f(s+c)$. A cue that activates across unrelated stems is more consistent with lexical/cue specificity; activation restricted to a subset of semantically related stems is more consistent with context-sensitive completion structure. This remains a diagnostic rather than a semantic label.

## Live candidate ranking and causal readiness

Large SAE coefficients can dominate a raw mean-difference ranking even when they are common across many concepts. The default candidate score is therefore

$$
S_f = \max(0,\mathrm{selectivity}_f) \cdot \mathrm{target\ rate}_f \cdot \log(1 + \mathrm{target\ mean}_f).
$$

This ranking is still exploratory. It is designed to triage candidates, not replace held-out AUROC/F1. The same forward pass optionally includes the current Workbench prompt, allowing the candidate table to report current-prompt maximum activation and selected-token activation. A candidate can therefore be concept-associated in the live batch but visibly inactive at the current causal location.

## v0.8 causal-ready candidate mode

Prompt-wide concept association and immediate causal usability are different constraints. A feature may rank highly for a concept across the controlled prompt set while having zero activation at the selected Workbench token, in which case ablation at that location is a no-op.

The **Causal-ready at current token** mode therefore requires:

```text
target_mean > 0
mean_difference > 0
current_token_activation > 0
```

and ranks eligible features using the balanced score multiplied by a log-scaled current-token activation term:

$$
S_f^{\mathrm{ready}} = S_f \cdot \log(1 + z_f^{\mathrm{current\ token}}).
$$

The log factor makes current-token presence matter without allowing a single very large coefficient to dominate as strongly as a raw activation product would. This remains an exploratory triage score, not held-out concept evidence.

## v0.8 cue-dominance summary

The cue × context matrix now summarizes the measured activation pattern rather than always returning generic interpretation text. For each cue, FeatureLens counts the number of tested contexts in which the feature is active and computes mean activation across those contexts.

A particularly strong tested cue-specific pattern occurs when one cue activates in every tested context and all other tested cues remain inactive. FeatureLens describes that pattern as **cue-dominant under the tested matrix**. The wording is intentionally local to the controlled stems and cues; it does not assert a universal semantic label for the SAE feature.

## v0.9 batched causal candidate triage

Concept-guided discovery and causal testing answer different questions. A feature can be selective for a controlled concept but have no activation at the selected Workbench token; conversely, a current-token-active feature may be causally irrelevant for the continuation under study.

v0.9 therefore inserts a low-cost triage stage between discovery and the full random-control causal test. For up to eight candidate features active or inactive at the current location, FeatureLens constructs the native ablation

$$
\Delta h_i = -z_i d_i
$$

for each candidate feature $i$, stacks a zero-edit reference plus all candidate deltas along the batch dimension, and teacher-forces the same target continuation for every condition in one model forward. The screen reports:

- native feature activation;
- perturbation L2 norm;
- target mean log-probability delta per token;
- target sequence log-probability delta;
- next-token Jensen-Shannon divergence.

Rows are ordered by absolute target mean-log-probability effect. This ordering is deliberately a **native-ablation effect screen**, not a significance or specificity statistic. No random-control ensemble is used at this stage. A promising candidate should be promoted to the existing single-feature causal test, which compares the SAE edit with the eight-direction norm-matched random ensemble.

This two-stage design reduces live GPU use while preserving the stronger causal standard for any result that is ultimately interpreted.


## v0.10 discovery-to-causality concordance

The central FeatureLens question is not only whether a concept-associated SAE feature can be intervened on, but whether **association strength predicts causal influence**. v0.10 therefore joins the live concept-discovery evidence with the batched candidate-ablation screen for the same shortlist.

For each screened feature, the synthesis keeps three orderings separate:

1. **Discovery rank** from the chosen exploratory concept-evidence score.
2. **Target-effect rank** from $|\Delta \bar{\ell}_{\mathrm{target}}|$, the absolute change in teacher-forced mean target log-probability per token under native ablation.
3. **Distribution-shift rank** from next-token Jensen–Shannon divergence.

The UI also reports a rank shift

$$
\Delta r_i = r_i^{\mathrm{discovery}} - r_i^{\mathrm{target}},
$$

so positive values indicate a feature that rises in the target-effect ordering relative to discovery, while negative values indicate a feature that looked stronger associatively than causally for the specified continuation.

Across the small screened set, FeatureLens computes tie-aware Spearman correlations between candidate score and (a) absolute target effect and (b) next-token JS. These are deliberately labeled **descriptive** because the live shortlist is small and the triage stage does not spend random-control ensembles. A promoted feature still requires the full single-feature causal test before specificity is interpreted.

This synthesis costs no additional model forward: it is computed from the discovery and triage tables already produced by the two-stage workflow.

## v0.11 controlled candidate specificity

The cheap candidate-ablation triage intentionally omits random controls so multiple candidate features can be screened in one small batch. Raw intervention magnitude is useful for triage, but it does not establish that an SAE direction is more behaviorally specific than an arbitrary residual perturbation of the same norm.

v0.11 therefore adds a second-stage controlled screen for at most three candidates. For candidate feature \(i\) with native activation \(z_i\) and decoder direction \(d_i\), the targeted ablation is

\[
\Delta h_i = -z_i d_i.
\]

For each candidate, FeatureLens generates `live_random_controls` deterministic random residual directions \(r_{ij}\) such that

\[
\|r_{ij}\|_2 = \|\Delta h_i\|_2.
\]

A single batched execution contains one zero-edit reference, every targeted SAE ablation, and every candidate-specific random control. All target and JS effects are therefore measured relative to the same batched null.

Two random-normalized causal quantities are kept separate:

1. **Target specificity ratio**

\[
\frac{|\Delta \bar{\ell}_{\text{SAE}}|}
     {\operatorname{mean}_j |\Delta \bar{\ell}_{r_j}|},
\]

where \(\bar{\ell}\) is the teacher-forced mean log probability per target token.

2. **JS specificity ratio**

\[
\frac{\operatorname{JS}(p_0, p_{\text{SAE}})}
     {\operatorname{mean}_j \operatorname{JS}(p_0, p_{r_j})}.
\]

The first asks whether the SAE edit is unusually influential for the specified continuation. The second asks whether it is unusually disruptive to the local next-token distribution as a whole. They are intentionally not collapsed into a single score.

The live empirical random-control tail uses the finite-ensemble correction

\[
p = \frac{1 + \#\{|e_{r_j}| \ge |e_{\text{SAE}}|\}}
         {1 + N_{\text{controls}}}.
\]

With eight controls, the smallest possible live value is \(1/9\approx0.111\). These values are therefore coarse specificity diagnostics, not conventional significance tests.

### Strategic controlled shortlist

After cheap triage, the default controlled shortlist is chosen to preserve disagreement rather than merely retest the top target-effect rows:

1. strongest discovery candidate among the screened features;
2. strongest raw target-effect candidate;
3. strongest next-token-JS candidate if distinct;
4. fill any remaining slot by target-effect rank.

This makes the controlled follow-up directly test the project's central question: whether strong concept-association evidence predicts random-normalized causal specificity.

### Association vs controlled causality

The zero-GPU synthesis layer joins discovery evidence to the controlled table and keeps four ranks separate:

- discovery rank;
- raw target-effect rank;
- random-normalized target-specificity rank;
- random-normalized JS-specificity rank.

Descriptive Spearman correlations between candidate score and the two specificity ratios are shown only as small-sample diagnostics. The full offline benchmark remains the place for larger candidate sets, more random controls, uncertainty intervals, and formal held-out conclusions.

## v0.12 split-half discovery stability and cross-target profiling

### Split-half discovery stability

The live concept-discovery batch already contains multiple controlled prompts per concept. v0.12 reuses those activations to form two prompt halves and independently reranks candidates under the same discovery mode. The UI reports the overlap/Jaccard of the two top-k sets. This costs no additional model inference.

Because the live setting uses only a few prompts per concept, this is a **sensitivity diagnostic**, not a confidence interval or reliability claim. Low overlap means the live shortlist is sample-sensitive and should not be treated as a stable semantic ranking.

### Controlled evidence patterns

For every random-controlled candidate, FeatureLens keeps target specificity and whole-distribution JS specificity separate. v0.12 adds a descriptive pattern layer:

- **Broad controlled influence**: both target and JS specificity ratios are at least 1.5× their matched-random means.
- **Target-weighted**: target specificity is at least 1.5× while JS specificity is lower.
- **Distribution-shift dominant**: JS specificity is at least 1.5× while target specificity is below 1×.
- **Distribution-shift weighted**: JS specificity is at least 1.5× while target specificity is weaker but not below 1×.
- **Weak / mixed specificity**: neither ratio clears the descriptive 1.5× threshold.

These are effect-ratio summaries only. The live eight-control empirical tails remain coarse and are not converted into significance labels.

### Cross-target causal profile

A feature can have a large effect on one specified continuation without being generally causal for the underlying concept. v0.12 therefore evaluates the same native SAE ablation against several exact continuations.

For each target continuation, FeatureLens computes a batched zero-edit baseline and reports:

- Δ mean log probability per target token,
- Δ full-sequence log probability,
- next-token JS divergence.

For each feature it also reports the strongest target by absolute mean-log-probability effect, the mean absolute effect on the remaining targets, and a **target-profile ratio**:

\[
R_{profile} = \frac{\max_t |\Delta \bar{\ell}_t|}{\operatorname{mean}_{u \ne t^*}|\Delta \bar{\ell}_u|}.
\]

This profile is a screening diagnostic and intentionally omits random controls. Random-normalized causal claims still require the Controlled candidate specificity experiment.

## v0.13 resample stability and pairwise target preference

### Deterministic balanced resampling

The live discovery forward already yields a prompt-wide SAE activation vector for every controlled prompt. v0.13 reuses that tensor for 32 deterministic balanced bootstrap resamples. Each concept is resampled independently with replacement so the live concept balance is preserved. Candidate ranking is recomputed under the currently selected ranking mode, including the fixed current-token compatibility term for `causal_ready`. For every candidate in the full-data shortlist, FeatureLens records the fraction of resamples in which it reappears and its median rank when present. No model or SAE forward is repeated. Because the live prompt count is deliberately tiny, these are sensitivity descriptors rather than statistical confidence estimates.

### Cross-target concentration

For one feature with target effects `Δ_t`, v0.13 normalizes `|Δ_t|` into a probability vector and reports normalized entropy `H / log(T)`, effect concentration `1 − H/log(T)`, and signed bias `ΣΔ_t / Σ|Δ_t|`. These quantities distinguish target-concentrated effects from broad effects and indicate whether broad influence is primarily suppressive, enhancing, or mixed-sign. The profile text is a descriptive heuristic only.

### Pairwise preference shifts

For every unordered target pair `(A, B)`, FeatureLens derives `Δ_pref(A,B) = Δmean(A) − Δmean(B)` from the already-computed teacher-forced mean-log-probability effects. This is the intervention-induced change in token-normalized preference for A relative to B. It does not add a model call and does not replace matched-random specificity controls.

## v0.14 offline-study methodology

### Prompt-wide SAE concept evidence

Offline concept-feature evaluation now max-pools each SAE feature across every non-padding token in a prompt. This aligns the held-out study with the live prompt-wide diagnostics and avoids assigning an entire prompt's concept evidence to an arbitrary final token. Separate final-token SAE matrices are still saved for local analyses.

### Activation-resample selection stability

`experiments/analyze_stability.py` performs deterministic balanced bootstrap resamples over the already-saved prompt-wide activation matrix. Within each resample, candidates are ranked using the live-compatible balanced score `positive selectivity × target activation rate × log1p(target mean)`. The output records shortlist support and resample-rank summaries. This is a sensitivity analysis, not a confidence interval.

### Study-level association vs causality

`experiments/analyze_study.py` selects one feature per concept using the original train-only AUROC discipline and joins held-out AUROC/F1 with paraphrase robustness, activation-resample support, causal-task activity, target-specificity, and JS-specificity. Random-normalized causal quantities are paired at the task level before aggregation. Cross-concept Spearman correlations use only seven concepts and are therefore reported descriptively.

### Reproducibility and artifact boundary

The full GPU + CPU pipeline supports `--resume`. Once expensive inference outputs exist, `experiments/run_analysis_only.py` reruns only CPU evaluation/stability/report stages. `scripts/validate_artifacts.py` enforces the public artifact schema and confirms that v0.14 prompt-wide activation metadata was used before results are surfaced in the public Offline study tab.
