# FeatureLens v0.15 validation

v0.15 is a **public-design and offline-runner hardening release**. It does not change the already validated live Qwen/SAE inference methods. Do not spend ZeroGPU quota rerunning causal, paraphrase, layer, cue, discovery, or feature-set experiments for this release.

## Local software gate

Run from the repository root:

```bash
python3 -m pytest -q && \
python3 -m compileall -q app.py featurelens experiments scripts && \
python3 -m ruff check app.py featurelens experiments tests scripts && \
python3 scripts/ui_smoke.py && \
python3 scripts/release_check.py
```

Then validate the Colab notebook JSON:

```bash
python3 - <<'PY'
import nbformat
nb = nbformat.read('notebooks/FeatureLens_Offline_Study_Colab.ipynb', as_version=4)
nbformat.validate(nb)
print('Colab notebook: PASS')
PY
```

## Hugging Face acceptance — no GPU calls

After pushing, only inspect the rendered interface.

### A. Header and navigation

Pass when:

- the header shows **FeatureLens** and one factual subtitle;
- there is no visible release/version badge;
- tabs read **Guide, Workbench, Feature sets, Features, Paraphrases, Layers, Study, Method**;
- tabs are visually flat rather than pill/card navigation.

### B. Guide

Open **Guide**.

Pass when:

- there is no three-card “step 1 / step 2 / step 3” onboarding grid;
- the workflow is short prose;
- headings use the serif display face while controls/body copy use the neutral sans-serif face;
- no gradients, glow, badge clusters, or decorative cards are visible.

### C. Workbench without running inference

Open **Workbench**.

Pass when:

- experiment sections have a clear typographic hierarchy;
- related fields sit close together and separate experiments have more breathing room;
- primary experiment buttons are compact muted-teal actions, not full-width desktop banners;
- **Copy TSV** is visually secondary;
- table titles are clearly larger than table body text;
- no duplicate context cards appear inside the tab—the global **Context** line is the context source of truth.

### D. Features tab without running inference

Open **Features**.

Pass when:

- discovery, triage, controlled comparison, cross-target profile, and feature diagnostics read as sections of one tool rather than nested cards;
- helper copy is short and does not repeatedly restate “not a semantic label” after every empty result;
- the page remains usable in both normal desktop width and a narrower browser window.

### E. Study empty state

Open **Study** before real artifacts are committed.

Pass when it clearly says the offline study is not materialized and does not show fabricated metrics or placeholder result plots.

## Colab runner dry check

Do not run the full model study merely to validate v0.15. Open the included notebook in Colab and execute only the first GPU-detection cell if desired. The full study should be started only when you are ready to produce the actual empirical artifacts.
