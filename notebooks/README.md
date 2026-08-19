# Notebooks

FeatureLens includes two Colab runners.

- `FeatureLens_Offline_Study_Colab.ipynb` — **canonical runner** for reproducing the complete study from a fresh checkout. It executes discovery/evaluation, both causal-position policies, feature-set diagnostics, study synthesis, and artifact validation.
- `FeatureLens_Causal_Addendum_Colab.ipynb` — migration/reproduction utility for a completed final-token baseline. It preserves the original baseline and computes only the max-feature-activation causal addendum plus CPU analysis.

For a fresh reproduction, use the full-study notebook. The addendum notebook is retained because it documents the exact path used to extend the original baseline without recomputing the expensive discovery activations.
