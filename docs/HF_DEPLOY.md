# Hugging Face deployment

FeatureLens targets a **Gradio SDK Space** with ZeroGPU hardware.

## Runtime shape

On Hugging Face, `FEATURELENS_EAGER_LOAD` defaults to `1`. The runtime loads:

- `Qwen/Qwen3-1.7B-Base`;
- Qwen-Scope SAE layers **4, 14, 26** only.

The live app does not need every SAE layer from the full repository.

`app.py` launches with `ssr_mode=False`, matching the deployment path that removed the earlier SSR/auth coroutine warning during Space testing.

## GPU-decorated actions

Current live actions include:

- concept-guided candidate feature discovery and candidate reuse;
- completion-cue sensitivity scans;
- bounded table/plot focus controls, descriptive PNG exports, and a persistent Workbench context banner;
- **Inspect sparse features**;
- **Run single-feature causal test**;
- **Run scale dose-response**;
- **Run contrastive preference test**;
- **Run joint feature-set causal test**;
- **Run 1/3/5-feature ablation sweep**;
- **Run individual-vs-joint decomposition**;
- **Inspect selected-feature geometry**;
- **Trace feature across prompt tokens**;
- **Run controlled concept contrast**;
- **Compare paraphrase representations**;
- **Compare layers**.

Allocation durations in `app.py` are ceilings requested from ZeroGPU, not expected wall-clock runtimes.

## Batch-first causal execution

FeatureLens deliberately batches related conditions so stronger diagnostics do not require a separate GPU callback for every condition.

Examples:

- scale dose-response stacks the six multipliers in one edited batch;
- single-feature causal tests stack zero edit, one targeted SAE edit, and eight norm-matched random controls;
- 1/3/5 feature-set sensitivity batches targeted edits and control ensembles;
- individual-vs-joint decomposition batches all individual ablations plus the joint ablation;
- controlled concept contrast evaluates its balanced prompt batch together;
- contrastive preference reuses one targeted delta/control ensemble while scoring the two exact continuations in two compact batched forwards.

The primary causal reference inside each experiment is a **batched zero-edit row**. This prevents batch-vs-single floating-point drift from being mistaken for an intervention effect.

## Greedy generation vs probability-level scoring

The single-feature causal test retains baseline and edited greedy generation because the visible text comparison is useful in a public demo.

The primary targeted causal metric uses teacher-forced **full-continuation** log-probability scoring. Greedy text may remain unchanged while probability-level metrics move.

The heavier feature-set, set-size, interaction, and contrast panels avoid unnecessary free-running generations.

## Clipboard export

Major output tables include a dedicated **Copy table with headers** action. The app serializes the result as tab-separated text before invoking the browser clipboard API. This makes pasted output self-describing and spreadsheet-friendly.

On successful clipboard write, the clicked button briefly changes to **✓ Copied with headers**. If browser clipboard permission is unavailable, the frontend uses a temporary-textarea fallback.

## Embedded-Space layout

The app uses:

- `gr.Blocks(fill_width=True)`;
- an explicitly centered desktop canvas up to 1600 px wide;
- restrained serif typography with normalized control/table sizes;
- consistent muted-teal action and copy buttons;
- bounded result-table heights;
- explicit bottom padding with no visible footer clutter;
- a browser-side ResizeObserver/MutationObserver that requests a resize reflow after dynamic result-height changes.

These changes reduce wasted horizontal space and mitigate the embedded-Space case where the outer page stopped extending after a large dynamic result. Hugging Face still owns the outer embedding frame, so compare with the direct `*.hf.space` URL if the parent page ever behaves differently.

## Offline benchmark

Do **not** run the complete research benchmark as an interactive public-Space action.

Run on separate CUDA compute:

```bash
python3 experiments/run_all.py
```

Both causal runners accept a configurable random-control count. For more stable offline control estimates, increase the value if compute permits, for example:

```bash
python3 -m experiments.run_causal --random-controls 16
python3 -m experiments.run_feature_sets --random-controls 16
```

Then commit only the small report/catalog/CSV/figure artifacts intended for presentation. Large activation arrays remain gitignored.

## Local UI launch smoke

Before pushing a release, run:

```bash
python3 scripts/ui_smoke.py
```

This opens the real Gradio `launch()` path on a temporary localhost port and immediately closes it. It exists specifically so theme/launch integration errors are caught before Hugging Face rebuilds the Space.
