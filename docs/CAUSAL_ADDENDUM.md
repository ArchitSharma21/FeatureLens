# v0.16 causal-position addendum

The completed v0.15 study used prompt-wide SAE evidence for concept discovery but intervened only at the final prompt token. v0.16 preserves that run as a baseline and adds a second causal policy.

## Position policies

### `final_token`

Patch the selected SAE feature at the final prompt token. This is the original v0.15 baseline.

### `max_feature_activation`

For the selected concept feature and causal prompt:

1. encode the selected layer at every non-padding prompt token;
2. read the selected SAE feature's TopK activation at each token;
3. choose the token with the largest activation;
4. apply the SAE edit and all norm-matched random controls at that same token.

The position is selected **only from SAE activation**. The target continuation, logits, and intervention effect are never used to choose the location.

If the selected feature is inactive everywhere in the prompt, the policy records zero coverage and uses the final token as a deterministic zero-delta fallback.

## Statistical unit

Ablation and 2× amplification are repeated interventions on the same causal prompt. v0.16 therefore uses the **causal task** as the primary paired inference unit:

- average the absolute SAE effect across ablation and amplification within each task;
- average each intervention's random-control ensemble, then average those random magnitudes within task;
- bootstrap/sign-flip the resulting one SAE-vs-random pair per causal task.

The report separately shows unconditional effects across all tasks and effects conditional on the selected feature being active at the intervention location.

## Addendum runner

With a completed v0.15 artifact directory:

```bash
python -m experiments.run_causal_addendum --resume
```

The runner:

1. migrates `causal_results.csv` to `causal_results_final_token.csv` without rerunning it;
2. computes `causal_results_max_active.csv`;
3. regenerates `causal_position_summary.csv`, study synthesis, figures, and report;
4. validates the finalized artifact schema.

It does **not** rerun activation collection, feature evaluation, candidate stability, or feature-set inference.
