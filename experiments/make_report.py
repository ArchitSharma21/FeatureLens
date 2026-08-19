from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import ARTIFACT_DIR
from featurelens.stats import (
    bootstrap_mean_ci,
    paired_bootstrap_difference_ci,
    paired_sign_flip_pvalue,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a truthful experiment report from saved metrics."
    )
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    return parser.parse_args()


def _selected_features(catalog: pd.DataFrame) -> pd.DataFrame:
    scored = catalog.copy()
    scored["activation_contrast"] = (
        scored["activation_rate_pos"] - scored["activation_rate_neg"]
    )
    ordered = scored.sort_values(
        ["concept", "train_auroc", "activation_contrast"],
        ascending=[True, False, False],
    )
    return ordered.groupby("concept", as_index=False).first()


def _effect_column(frame: pd.DataFrame) -> str:
    if "target_mean_logprob_delta" in frame.columns:
        return "target_mean_logprob_delta"
    return "target_logprob_delta"


def _causal_file(artifact_dir: Path, policy: str) -> Path:
    if policy == "final_token":
        name = "causal_results_final_token.csv"
    else:
        name = "causal_results_max_active.csv"

    path = artifact_dir / name
    if path.exists():
        return path

    legacy = artifact_dir / "causal_results.csv"
    if policy == "final_token" and legacy.exists():
        return legacy
    return path


def _task_level_stats(
    frame: pd.DataFrame,
    *,
    seed: int,
    active_only: bool = False,
) -> dict:
    metric = _effect_column(frame)
    sae_rows = frame[frame["condition"] == "sae_feature"].copy()

    if active_only:
        if "feature_active_at_intervention" in frame.columns:
            active_col = "feature_active_at_intervention"
        else:
            active_col = "feature_activation"

        active_mask = (
            pd.to_numeric(sae_rows[active_col], errors="coerce").fillna(0) > 0
        )
        active_ids = sae_rows.loc[active_mask, "task_id"].unique()
        frame = frame[frame["task_id"].isin(active_ids)].copy()
        sae_rows = frame[frame["condition"] == "sae_feature"].copy()

    sae = (
        sae_rows.assign(
            _abs=lambda data: np.abs(
                pd.to_numeric(data[metric], errors="coerce")
            )
        )
        .groupby("task_id", as_index=False)
        .agg(sae_abs=("_abs", "mean"))
    )
    random = (
        frame[frame["condition"] == "random_norm_matched"]
        .assign(
            _abs=lambda data: np.abs(
                pd.to_numeric(data[metric], errors="coerce")
            )
        )
        .groupby(["task_id", "intervention"], as_index=False)["_abs"]
        .mean()
        .groupby("task_id", as_index=False)["_abs"]
        .mean()
        .rename(columns={"_abs": "random_abs"})
    )
    paired = sae.merge(random, on="task_id", how="inner")

    if paired.empty:
        nan = float("nan")
        return {
            "sae_abs": nan,
            "random_abs": nan,
            "ratio": nan,
            "advantage": nan,
            "ci": [nan, nan],
            "pvalue": nan,
            "n_tasks": 0,
        }

    sae_effect = paired["sae_abs"].to_numpy(float)
    random_effect = paired["random_abs"].to_numpy(float)
    low, high = paired_bootstrap_difference_ci(
        sae_effect,
        random_effect,
        seed=seed,
    )
    return {
        "sae_abs": float(sae_effect.mean()),
        "random_abs": float(random_effect.mean()),
        "ratio": float(
            sae_effect.mean() / max(float(random_effect.mean()), 1e-12)
        ),
        "advantage": float((sae_effect - random_effect).mean()),
        "ci": [float(low), float(high)],
        "pvalue": float(
            paired_sign_flip_pvalue(
                sae_effect,
                random_effect,
                seed=seed + 1,
            )
        ),
        "n_tasks": int(len(paired)),
    }


def _paired_stats(
    frame: pd.DataFrame,
    *,
    index: list[str],
    sae_condition: str,
    random_condition: str,
    seed: int,
) -> dict[str, float | list[float]]:
    """Legacy helper retained for report-control regression tests."""
    metric = _effect_column(frame)
    sae = (
        frame[frame["condition"] == sae_condition]
        .groupby(index, as_index=False)[metric]
        .first()
        .rename(columns={metric: "sae_effect"})
    )
    random = (
        frame[frame["condition"] == random_condition]
        .assign(
            _abs_effect=lambda data: np.abs(
                pd.to_numeric(data[metric], errors="coerce")
            )
        )
        .groupby(index, as_index=False)["_abs_effect"]
        .mean()
        .rename(columns={"_abs_effect": "random_abs_effect"})
    )
    paired = sae.merge(random, on=index, how="inner")
    sae_effect = np.abs(paired["sae_effect"].to_numpy(dtype=float))
    random_effect = paired["random_abs_effect"].to_numpy(dtype=float)

    if sae_effect.size == 0:
        nan = float("nan")
        return {
            "sae_abs": nan,
            "random_abs": nan,
            "ratio": nan,
            "paired_advantage": nan,
            "ci": [nan, nan],
            "pvalue": nan,
            "n_pairs": 0,
        }

    low, high = paired_bootstrap_difference_ci(
        sae_effect,
        random_effect,
        seed=seed,
    )
    return {
        "sae_abs": float(sae_effect.mean()),
        "random_abs": float(random_effect.mean()),
        "ratio": float(
            sae_effect.mean() / max(float(random_effect.mean()), 1e-12)
        ),
        "paired_advantage": float((sae_effect - random_effect).mean()),
        "ci": [float(low), float(high)],
        "pvalue": float(
            paired_sign_flip_pvalue(
                sae_effect,
                random_effect,
                seed=seed + 1,
            )
        ),
        "n_pairs": int(sae_effect.size),
    }


def _feature_set_stats(frame: pd.DataFrame, *, seed: int) -> dict:
    metric = _effect_column(frame)
    sae = (
        frame[frame["condition"] == "sae_feature_set"]
        .groupby("task_id", as_index=False)[metric]
        .first()
        .rename(columns={metric: "sae"})
    )
    random = (
        frame[frame["condition"] == "random_norm_matched"]
        .assign(
            _abs=lambda data: np.abs(
                pd.to_numeric(data[metric], errors="coerce")
            )
        )
        .groupby("task_id", as_index=False)["_abs"]
        .mean()
        .rename(columns={"_abs": "random"})
    )
    paired = sae.merge(random, on="task_id")
    sae_effect = np.abs(paired["sae"].to_numpy(float))
    random_effect = paired["random"].to_numpy(float)

    if sae_effect.size == 0:
        return {}

    low, high = paired_bootstrap_difference_ci(
        sae_effect,
        random_effect,
        seed=seed,
    )
    return {
        "sae_abs": float(sae_effect.mean()),
        "random_abs": float(random_effect.mean()),
        "ratio": float(
            sae_effect.mean() / max(float(random_effect.mean()), 1e-12)
        ),
        "advantage": float((sae_effect - random_effect).mean()),
        "ci": [float(low), float(high)],
        "pvalue": float(
            paired_sign_flip_pvalue(
                sae_effect,
                random_effect,
                seed=seed + 1,
            )
        ),
        "n_tasks": int(sae_effect.size),
    }


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_plots(
    artifact_dir: Path,
    selected: pd.DataFrame,
    layers: pd.DataFrame,
    max_active: pd.DataFrame,
    feature_sets: pd.DataFrame | None,
    position: pd.DataFrame,
    study: pd.DataFrame,
) -> None:
    fig_dir = artifact_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.5, 4.2))
    ax = fig.add_subplot(111)
    ordered = selected.sort_values("auroc")
    ax.barh(ordered["concept"], ordered["auroc"])
    ax.axvline(0.5, linewidth=1, linestyle="--")
    ax.set_xlabel("Held-out AUROC")
    ax.set_title("Selected SAE feature predictiveness")
    _save_figure(fig, fig_dir / "feature_auroc.png")

    fig = plt.figure(figsize=(7.0, 4.2))
    ax = fig.add_subplot(111)
    ax.plot(
        layers["layer"],
        layers["linear_probe_macro_auroc"],
        marker="o",
        label="Linear probe AUROC",
    )
    ax.plot(
        layers["layer"],
        layers["reconstruction_cosine"],
        marker="o",
        label="SAE reconstruction cosine",
    )
    ax.set_xlabel("Layer")
    ax.set_ylim(0, 1.05)
    ax.set_title("Layer-wise representation diagnostics")
    ax.legend()
    _save_figure(fig, fig_dir / "layer_diagnostics.png")

    metric = _effect_column(max_active)
    grouped = (
        max_active.groupby(["intervention", "condition"])[metric]
        .apply(lambda values: float(np.mean(np.abs(values))))
        .reset_index(name="mean_abs_effect")
    )
    pivot = grouped.pivot(
        index="intervention",
        columns="condition",
        values="mean_abs_effect",
    )
    fig = plt.figure(figsize=(7, 4.2))
    ax = fig.add_subplot(111)
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean |Δ mean log p/token|")
    ax.set_title("Max-active SAE edits vs norm-matched controls")
    ax.tick_params(axis="x", rotation=0)
    _save_figure(fig, fig_dir / "causal_effects.png")

    if feature_sets is not None and not feature_sets.empty:
        feature_set_metric = _effect_column(feature_sets)
        grouped_sets = (
            feature_sets.groupby(["set_size", "condition"])[feature_set_metric]
            .apply(lambda values: float(np.mean(np.abs(values))))
            .reset_index(name="mean_abs_effect")
        )
        set_pivot = grouped_sets.pivot(
            index="set_size",
            columns="condition",
            values="mean_abs_effect",
        )
        fig = plt.figure(figsize=(7, 4.2))
        ax = fig.add_subplot(111)
        set_pivot.plot(kind="line", marker="o", ax=ax)
        ax.set_xlabel("Jointly ablated feature count")
        ax.set_ylabel("Mean |Δ mean log p/token|")
        ax.set_title("Final-token feature-set diagnostic")
        _save_figure(fig, fig_dir / "feature_set_effects.png")

    overall = position[position["concept"] == "__all__"].copy()
    policy_order = ["final_token", "max_feature_activation"]
    overall["position_policy"] = pd.Categorical(
        overall["position_policy"],
        categories=policy_order,
        ordered=True,
    )
    overall = overall.sort_values("position_policy")
    position_plot = pd.DataFrame(
        {
            "Policy": ["Final token", "Max feature activation"],
            "SAE effect": overall["target_sae_abs_mean"].to_numpy(float),
            "Random control": overall["target_random_abs_mean"].to_numpy(float),
        }
    )
    fig = plt.figure(figsize=(7.2, 4.4))
    ax = fig.add_subplot(111)
    x_positions = np.arange(len(position_plot))
    width = 0.34
    ax.bar(
        x_positions - width / 2,
        position_plot["SAE effect"],
        width,
        label="SAE effect",
    )
    ax.bar(
        x_positions + width / 2,
        position_plot["Random control"],
        width,
        label="Random control",
    )
    ax.set_xticks(x_positions, position_plot["Policy"])
    ax.set_ylabel("Task-level mean |Δ mean log p/token|")
    ax.set_title("Causal position sensitivity")
    ax.legend()
    _save_figure(fig, fig_dir / "causal_position_sensitivity.png")

    if not study.empty:
        fig = plt.figure(figsize=(7.2, 4.6))
        ax = fig.add_subplot(111)
        ax.scatter(
            study["heldout_auroc"],
            study["max_active_target_specificity_ratio"],
        )
        for row in study.itertuples():
            ax.annotate(
                str(row.concept),
                (row.heldout_auroc, row.max_active_target_specificity_ratio),
                fontsize=8,
            )
        ax.set_xlabel("Held-out feature AUROC")
        ax.set_ylabel("Max-active target specificity ratio")
        ax.set_title("Association evidence vs max-active causality")
        _save_figure(fig, fig_dir / "association_vs_causality.png")


def _coverage(
    frame: pd.DataFrame,
    column: str,
    fallback: str = "feature_activation",
) -> float:
    name = column if column in frame.columns else fallback
    values = pd.to_numeric(frame[name], errors="coerce").fillna(0)
    return float((values > 0).mean())


def _build_interpretation(
    *,
    max_stats: dict,
    final_coverage: float,
    max_coverage: float,
) -> str:
    strong = (
        max_stats["ratio"] >= 1.5
        and max_stats["ci"][0] > 0
        and max_stats["pvalue"] < 0.05
    )
    if strong:
        interpretation = (
            "Max-active interventions produced larger task-level target effects than "
            "norm-matched random controls with paired uncertainty excluding zero. "
            "Predictive SAE features therefore show causal specificity when intervened "
            "where the selected feature is actually represented, while the final-token "
            "baseline quantifies sensitivity to intervention location."
        )
    elif max_stats["ratio"] >= 1.5:
        interpretation = (
            "Max-active interventions had a larger point-estimate effect than norm-matched "
            "random controls, but task-level paired uncertainty did not support a strong "
            "significance claim. The result is therefore reported as suggestive causal "
            "specificity rather than conclusive evidence."
        )
    else:
        interpretation = (
            "Held-out feature predictiveness was strong, but max-active causal effects were "
            "only modest relative to norm-matched random controls. FeatureLens therefore "
            "separates predictive association from causal control rather than treating them "
            "as interchangeable."
        )

    if max_coverage > final_coverage + 0.1:
        interpretation += (
            " Moving from the final prompt token to the feature's maximum-activation token "
            f"increased intervention coverage from {final_coverage:.1%} to "
            f"{max_coverage:.1%}, showing that causal conclusions depend materially on "
            "where the representation is tested."
        )
    return interpretation


def _build_report_lines(
    *,
    headline: str,
    interpretation: str,
    highlights: list[str],
    feature_sets: pd.DataFrame | None,
    study: pd.DataFrame,
) -> list[str]:
    lines = [
        "# FeatureLens experiment report",
        "",
        "## Research question",
        "",
        "**Do sparse features that predict a concept also causally influence model behaviour?**",
        "",
        "## Executive summary",
        "",
        headline,
        "",
        interpretation,
        "",
        "## Key measurements",
        "",
        *[f"- {item}" for item in highlights],
        "",
        "## Experimental design",
        "",
        "- Model: Qwen3-1.7B-Base.",
        "- SAEs: Qwen-Scope residual-stream TopK SAEs at configured early/middle/late layers.",
        "- Discovery evidence: prompt-wide maximum SAE activation across non-padding tokens; final-token activations are saved separately.",
        "- Split discipline: paraphrase groups remain entirely in train or held-out test.",
        "- Feature selection: training-split AUROC plus activation contrast; held-out AUROC/F1 are reported separately.",
        "- Causal position policies: final prompt token and maximum selected-feature activation within the prompt. Max-active positions are selected from SAE activation only, never from behavioral outcomes.",
        "- Primary causal statistical unit: causal task. Ablation and 2× amplification are averaged within task before paired bootstrap/sign-flip inference.",
        "- Negative control: deterministic norm-matched random residual directions.",
        "- Primary target metric: exact full continuation mean log probability per token under teacher forcing.",
        "- Coverage and conditional-on-active effect strength are reported separately.",
        "- Feature-set analysis remains a final-token diagnostic and is not conflated with the max-active single-feature study.",
        "",
        "## Figures",
        "",
        "![Feature AUROC](figures/feature_auroc.png)",
        "",
        "![Layer diagnostics](figures/layer_diagnostics.png)",
        "",
        "![Causal position sensitivity](figures/causal_position_sensitivity.png)",
        "",
        "![Max-active causal effects](figures/causal_effects.png)",
    ]

    if feature_sets is not None and not feature_sets.empty:
        lines.extend(
            [
                "",
                "![Feature-set diagnostic](figures/feature_set_effects.png)",
            ]
        )

    if not study.empty:
        lines.extend(
            [
                "",
                "![Association vs causality](figures/association_vs_causality.png)",
                "",
                "## Position sensitivity",
                "",
                "The final-token policy asks whether the selected feature matters at the conventional last-prompt-token intervention site. The max-active policy asks whether it matters where that same feature is most strongly represented in the prompt. Reporting both prevents low final-token coverage from being mistaken for evidence that a predictive feature is globally non-causal.",
                "",
                "## Association vs causality across concepts",
                "",
                "Cross-concept correlations use max-active random-normalized specificity and are descriptive because the study has seven controlled concepts.",
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "High held-out AUROC is correlational evidence. Causal claims require downstream changes relative to norm-matched random controls. Max-active positions are chosen without reference to behavioral effect size. Task-level uncertainty treats ablation and amplification on the same causal prompt as repeated interventions, not independent experimental units.",
            "",
            "## Reproducibility",
            "",
            "Run `python -m experiments.run_all --resume` for a fresh full study. For an existing v0.15 final-token study, run the v0.16 causal addendum notebook; it preserves the baseline, computes only max-active causal rows, and reruns CPU analysis/reporting.",
            "",
        ]
    )
    return lines


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir

    catalog = pd.read_csv(artifact_dir / "feature_catalog.csv")
    layers = pd.read_csv(artifact_dir / "layer_metrics.csv")
    stability = pd.read_csv(artifact_dir / "stability.csv")
    final = pd.read_csv(_causal_file(artifact_dir, "final_token"))
    max_active = pd.read_csv(
        _causal_file(artifact_dir, "max_feature_activation")
    )

    feature_set_path = artifact_dir / "feature_set_results.csv"
    feature_sets = (
        pd.read_csv(feature_set_path) if feature_set_path.exists() else None
    )

    study_path = artifact_dir / "study_feature_summary.csv"
    study = pd.read_csv(study_path) if study_path.exists() else pd.DataFrame()
    position = pd.read_csv(artifact_dir / "causal_position_summary.csv")

    study_summary_path = artifact_dir / "study_summary.json"
    if study_summary_path.exists():
        study_summary = json.loads(study_summary_path.read_text())
    else:
        study_summary = {}

    selected = _selected_features(catalog)
    _save_plots(
        artifact_dir,
        selected,
        layers,
        max_active,
        feature_sets,
        position,
        study,
    )

    mean_auc = float(selected["auroc"].mean())
    median_auc = float(selected["auroc"].median())
    auc_low, auc_high = bootstrap_mean_ci(
        selected["auroc"].to_numpy(),
        seed=42,
    )

    best = layers.sort_values(
        "linear_probe_macro_auroc",
        ascending=False,
    ).iloc[0]
    mean_jaccard = float(stability["topk_jaccard"].mean())
    mean_cosine = float(stability["sparse_cosine"].mean())

    final_stats = _task_level_stats(final, seed=43)
    final_active_stats = _task_level_stats(
        final,
        seed=44,
        active_only=True,
    )
    max_stats = _task_level_stats(max_active, seed=45)
    max_active_stats = _task_level_stats(
        max_active,
        seed=46,
        active_only=True,
    )

    final_sae = final[final["condition"] == "sae_feature"]
    max_sae = max_active[max_active["condition"] == "sae_feature"]
    final_coverage = _coverage(
        final_sae,
        "feature_active_at_intervention",
    )
    anywhere_coverage = _coverage(
        max_sae,
        "feature_active_anywhere",
    )
    max_coverage = _coverage(
        max_sae,
        "feature_active_at_intervention",
    )

    set_summary: dict[int, dict] = {}
    if feature_sets is not None and not feature_sets.empty:
        sizes = sorted(int(value) for value in feature_sets["set_size"].unique())
        for size in sizes:
            subset = feature_sets[feature_sets["set_size"] == size]
            set_summary[size] = _feature_set_stats(
                subset,
                seed=100 + size,
            )

    interpretation = _build_interpretation(
        max_stats=max_stats,
        final_coverage=final_coverage,
        max_coverage=max_coverage,
    )

    headline = (
        f"Selected SAE features averaged {mean_auc:.3f} held-out AUROC. "
        f"Max-active interventions covered {max_coverage:.1%} of causal tasks and "
        f"changed mean log p/token by {max_stats['sae_abs']:.3f} in absolute value "
        f"on average versus {max_stats['random_abs']:.3f} for norm-matched random "
        f"controls ({max_stats['ratio']:.2f}×)."
    )

    highlights = [
        (
            f"Median selected-feature held-out AUROC: {median_auc:.3f}; mean AUROC "
            f"95% bootstrap CI [{auc_low:.3f}, {auc_high:.3f}]."
        ),
        (
            f"Best residual linear-probe layer: {int(best['layer'])} with macro AUROC "
            f"{float(best['linear_probe_macro_auroc']):.3f}."
        ),
        (
            f"Mean paraphrase TopK Jaccard: {mean_jaccard:.3f}; sparse activation "
            f"cosine: {mean_cosine:.3f}."
        ),
        (
            f"Feature coverage: final-token policy {final_coverage:.1%}; active "
            f"anywhere in prompt {anywhere_coverage:.1%}; max-active intervention "
            f"{max_coverage:.1%}."
        ),
        (
            f"Final-token task-level SAE/random ratio: {final_stats['ratio']:.2f}×; "
            f"paired advantage {final_stats['advantage']:+.4f}, 95% CI "
            f"[{final_stats['ci'][0]:+.4f}, {final_stats['ci'][1]:+.4f}], "
            f"sign-flip p={final_stats['pvalue']:.4f}."
        ),
        (
            f"Max-active task-level SAE/random ratio: {max_stats['ratio']:.2f}×; "
            f"paired advantage {max_stats['advantage']:+.4f}, 95% CI "
            f"[{max_stats['ci'][0]:+.4f}, {max_stats['ci'][1]:+.4f}], "
            f"sign-flip p={max_stats['pvalue']:.4f}."
        ),
        (
            "Conditional on feature-active tasks, max-active SAE/random ratio: "
            f"{max_active_stats['ratio']:.2f}× "
            f"(n={max_active_stats['n_tasks']})."
        ),
    ]

    if set_summary:
        max_size = max(set_summary)
        set_stats = set_summary[max_size]
        highlights.append(
            f"Final-token top-{max_size} joint ablation SAE/random ratio: "
            f"{set_stats['ratio']:.2f}×; paired advantage "
            f"{set_stats['advantage']:+.4f}, 95% CI "
            f"[{set_stats['ci'][0]:+.4f}, {set_stats['ci'][1]:+.4f}], "
            f"sign-flip p={set_stats['pvalue']:.4f}."
        )

    correlations = study_summary.get("correlations", {})
    target_corr = correlations.get(
        "heldout_auroc_vs_max_active_target_specificity",
        {},
    )
    js_corr = correlations.get(
        "heldout_auroc_vs_max_active_js_specificity",
        {},
    )
    if target_corr:
        target_rho = float(target_corr.get("rho", float("nan")))
        js_rho = float(js_corr.get("rho", float("nan")))
        highlights.extend(
            [
                (
                    "Across seven concepts, held-out AUROC vs max-active target "
                    f"specificity Spearman ρ={target_rho:+.3f}; descriptive only."
                ),
                (
                    "Held-out AUROC vs max-active JS specificity Spearman "
                    f"ρ={js_rho:+.3f}; descriptive only."
                ),
            ]
        )

    summary = {
        "headline": headline,
        "highlights": highlights,
        "interpretation": interpretation,
        "metrics": {
            "mean_selected_feature_test_auroc": mean_auc,
            "mean_selected_feature_test_auroc_bootstrap_ci_95": [
                auc_low,
                auc_high,
            ],
            "median_selected_feature_test_auroc": median_auc,
            "best_linear_probe_layer": int(best["layer"]),
            "best_linear_probe_macro_auroc": float(
                best["linear_probe_macro_auroc"]
            ),
            "mean_paraphrase_topk_jaccard": mean_jaccard,
            "mean_paraphrase_sparse_cosine": mean_cosine,
            "final_token_feature_coverage": final_coverage,
            "prompt_anywhere_feature_coverage": anywhere_coverage,
            "max_active_feature_coverage": max_coverage,
            "final_token_task_level": final_stats,
            "final_token_active_only": final_active_stats,
            "max_active_task_level": max_stats,
            "max_active_active_only": max_active_stats,
            "feature_set_results": {
                str(key): value for key, value in set_summary.items()
            },
            "study_summary": study_summary,
        },
    }
    (artifact_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    report_lines = _build_report_lines(
        headline=headline,
        interpretation=interpretation,
        highlights=highlights,
        feature_sets=feature_sets,
        study=study,
    )
    report_path = artifact_dir / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(headline)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
