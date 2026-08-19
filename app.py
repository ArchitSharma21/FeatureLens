from __future__ import annotations

import gradio as gr
import pandas as pd

from featurelens.config import SETTINGS
from featurelens.hf_runtime import gpu
from featurelens.runtime import RUNTIME
from featurelens.study import OfflineStudy

# Restrained, print-inspired palette. The app deliberately avoids saturated dashboard colors.
INK_TEAL = "#6F8984"
INK_UMBER = "#8A735D"
INK_RED = "#8C6A67"
INK_PLUM = "#786F82"
INK_STONE = "#82827E"
INK_BLUEGREY = "#687982"

LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
]

STUDY = OfflineStudy()

CSS = r"""
/*
FeatureLens UI system
---------------------
This is an analytical instrument, not a SaaS landing page. The interface uses
plain surfaces, a restrained accent, strong typographic hierarchy, compact
forms, and deliberate spacing instead of nested cards, badges, glow, or
uniform full-width CTAs.
*/
.gradio-container {
  --fl-accent: #6F8984;
  --fl-accent-hover: #607A75;
  --fl-rule: var(--border-color-primary);
  --fl-body: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  --fl-display: Georgia, Cambria, "Times New Roman", serif;
  --fl-mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  width: min(95vw, 1500px) !important;
  max-width: 1500px !important;
  margin: 0 auto !important;
  padding: 0 24px 112px !important;
  font-family: var(--fl-body) !important;
  font-size: 15.5px !important;
  line-height: 1.48;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container button,
.gradio-container select,
.gradio-container label,
.gradio-container table,
.gradio-container .prose {
  font-family: var(--fl-body) !important;
}
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container .section-rule,
.gradio-container .table-heading,
.gradio-container .hero-title {
  font-family: var(--fl-display) !important;
}
.gradio-container p,
.gradio-container li { font-size: 15.5px; }
.gradio-container h2 { font-size: 1.68rem; line-height: 1.22; margin-bottom: .45rem; }
.gradio-container h3 { font-size: 1.36rem; line-height: 1.26; margin-bottom: .4rem; }
.gradio-container h4 { font-size: 1.14rem; line-height: 1.30; margin-bottom: .35rem; }

/* Header: compact, editorial, no product-release chrome. */
.hero {
  padding: 18px 0 14px;
  border-bottom: 1px solid var(--fl-rule);
  margin-bottom: 10px;
}
.hero h1 {
  margin: 0;
  font-family: var(--fl-display) !important;
  font-size: 2rem;
  font-weight: 600;
  line-height: 1.05;
  letter-spacing: -.015em;
}
.hero .subtitle {
  margin-top: 6px;
  max-width: 68ch;
  font-size: .98rem;
  color: var(--body-text-color-subdued);
}

/* Navigation is deliberately flat: text tabs + one active rule. */
.gradio-container .tabs > .tab-nav,
.gradio-container [role="tablist"] {
  gap: 2px !important;
  border-bottom: 1px solid var(--fl-rule) !important;
}
.gradio-container [role="tab"] {
  border-radius: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  font-weight: 500 !important;
  padding: 9px 11px !important;
}
.gradio-container [role="tab"][aria-selected="true"] {
  color: var(--body-text-color) !important;
  border-bottom: 2px solid var(--fl-accent) !important;
}

/* Current context is a quiet status line, not a side-accent card. */
.context-card {
  border: 0 !important;
  border-bottom: 1px solid var(--fl-rule) !important;
  background: transparent !important;
  padding: 8px 0 10px !important;
  margin: 0 0 14px !important;
  border-radius: 0 !important;
  color: var(--body-text-color-subdued);
}
.context-card p { margin: 0 !important; font-size: .93rem !important; }

/* Introductory prose stays narrow even when the analytical canvas is wide. */
.guide-intro { max-width: 72ch; margin-bottom: 10px; }

/* Section rhythm: no roman numerals, small-caps, or decorative top rules. */
.section-rule {
  margin: 28px 0 6px;
  padding: 0;
  border: 0;
  font-variant: normal;
  letter-spacing: 0;
  font-size: 1.34rem;
  font-weight: 600;
  opacity: 1;
}
.section-note,
.candidate-help,
.form-note,
.graph-note,
.small-note {
  max-width: 78ch;
  color: var(--body-text-color-subdued) !important;
  opacity: 1 !important;
  font-size: .91rem !important;
}
.section-note { margin: 0 0 12px !important; }
.form-note { margin: -2px 0 7px !important; }
.candidate-help { margin-top: 2px !important; }

/* Explanatory callouts use a plain rule instead of card / side-tab styling. */
.instrument-note {
  border: 0 !important;
  border-top: 1px solid var(--fl-rule) !important;
  border-bottom: 1px solid var(--fl-rule) !important;
  border-radius: 0 !important;
  padding: 8px 0 !important;
  background: transparent !important;
  margin: 4px 0 12px !important;
  color: var(--body-text-color-subdued);
  font-size: .92rem;
}

/* Forms stay utilitarian and square-ish. */
.gradio-container .form,
.gradio-container .block { border-radius: 2px !important; }
.gradio-container .group {
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin: 0 !important;
}
.gradio-container textarea,
.gradio-container input,
.gradio-container select { border-radius: 2px !important; font-size: 15px !important; }
.gradio-container label,
.gradio-container .label-wrap { font-size: 14.5px !important; font-weight: 500 !important; }

/* Primary actions are compact; utilities are quiet and secondary. */
.gradio-container button {
  border-radius: 2px !important;
  font-size: 14.5px !important;
  font-weight: 600 !important;
  box-shadow: none !important;
}
.action-btn { width: fit-content !important; max-width: 100% !important; }
.action-btn button {
  width: auto !important;
  min-height: 36px !important;
  padding: 7px 14px !important;
  background: var(--fl-accent) !important;
  color: #fff !important;
  border: 1px solid var(--fl-accent) !important;
}
.action-btn button:hover {
  background: var(--fl-accent-hover) !important;
  border-color: var(--fl-accent-hover) !important;
}
.copy-btn { width: fit-content !important; max-width: 100% !important; margin-top: 3px !important; }
.copy-btn button {
  width: auto !important;
  min-height: 31px !important;
  padding: 5px 10px !important;
  background: transparent !important;
  color: var(--body-text-color-subdued) !important;
  border: 1px solid var(--fl-rule) !important;
  font-weight: 500 !important;
}
.copy-btn button:hover {
  background: var(--background-fill-secondary) !important;
  color: var(--body-text-color) !important;
}

/* Tokens are data, so monospace is appropriate here and nowhere else. */
.token-wrap { display: flex; flex-wrap: wrap; gap: 5px; padding: 6px 0 10px; line-height: 1.85; }
.token {
  background: var(--background-fill-secondary);
  border: 1px solid var(--fl-rule);
  border-radius: 2px;
  padding: 2px 6px;
  font-family: var(--fl-mono) !important;
  font-size: 12px;
}
.token.selected { border-color: var(--fl-accent); outline: 1px solid var(--fl-accent); font-weight: 600; }
.token sup { opacity: .58; margin-right: 4px; }

/* Result tables: normal-weight data, tabular numerics, clear heading. */
.table-heading {
  margin: 2px 0 -32px !important;
  padding: 4px 58px 0 0 !important;
  min-height: 32px;
  position: relative;
  z-index: 3;
  pointer-events: none;
  font-size: 1.16rem !important;
  font-weight: 600 !important;
  line-height: 1.22 !important;
}
.result-table table,
.result-table [role="grid"] {
  font-family: var(--fl-body) !important;
  font-size: 14.25px !important;
  font-variant-numeric: tabular-nums;
}
.result-table table thead th,
.result-table table thead th *,
.result-table [role="columnheader"],
.result-table [role="columnheader"] * {
  font-size: 14.25px !important;
  font-weight: 650 !important;
  line-height: 1.24 !important;
}
.result-table table tbody td,
.result-table [role="gridcell"] {
  font-size: 14.25px !important;
  font-weight: 400 !important;
  line-height: 1.32 !important;
}
.result-table .label-wrap,
.result-table [data-testid="block-label"],
.result-table .block-label,
.result-table .block-title { display: none !important; }

/* Plots use the body face for axes/legends; titles can retain their chart style. */
.fl-plot svg text { font-family: var(--fl-body) !important; }

/* Existing in-place focus behavior is preserved exactly. */
.fl-plot.featurelens-inline-focus,
.result-table.featurelens-inline-focus {
  position: relative !important;
  z-index: 5000 !important;
  background: var(--background-fill-primary) !important;
  border: 1px solid var(--fl-rule) !important;
  box-shadow: 0 10px 28px rgba(0, 0, 0, .34) !important;
  border-radius: 2px !important;
}
.fl-plot.featurelens-inline-focus { transform-origin: top left !important; }
.result-table.featurelens-inline-focus { overflow: visible !important; }

.wide-table { width: 100% !important; }
.bottom-spacer { height: 86px; width: 100%; }
.tabs, .tabitem { padding-bottom: 22px !important; }

/* Keep prose readable instead of spanning the full analytical canvas. */
.prose p,
.prose li { max-width: 80ch; }

@media (max-width: 900px) {
  .gradio-container { width: 100% !important; padding-left: 12px !important; padding-right: 12px !important; }
  .action-btn, .copy-btn { width: 100% !important; }
  .action-btn button, .copy-btn button { width: 100% !important; }
}
"""
THEME = gr.themes.Base(
    primary_hue="teal",
    secondary_hue="gray",
    neutral_hue="gray",
    radius_size="sm",
)

COPY_JS = r"""
(text) => {
  const value = text || "";
  const button = document.activeElement && document.activeElement.tagName === "BUTTON"
    ? document.activeElement : null;
  const oldLabel = button ? button.innerText : null;
  const signal = () => {
    if (!button) return;
    button.innerText = "Copied";
    button.disabled = true;
    window.setTimeout(() => {
      button.innerText = oldLabel || "Copy TSV";
      button.disabled = false;
    }, 1200);
  };
  const fallback = () => {
    const node = document.createElement("textarea");
    node.value = value;
    node.style.position = "fixed";
    node.style.opacity = "0";
    document.body.appendChild(node);
    node.focus();
    node.select();
    document.execCommand("copy");
    document.body.removeChild(node);
    signal();
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(value).then(signal).catch(fallback);
  } else {
    fallback();
  }
  return [value];
}
"""

INSTALL_REFLOW_JS = r"""
() => {
  if (window.__featurelens_reflow_installed) return [];
  window.__featurelens_reflow_installed = true;
  let timer = null;
  const kick = () => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => window.dispatchEvent(new Event("resize")), 80);
  };
  const root = document.querySelector(".gradio-container") || document.body;
  if (window.ResizeObserver) {
    const observer = new ResizeObserver(kick);
    observer.observe(root);
    window.__featurelens_reflow_observer = observer;
  }
  const mutation = new MutationObserver(kick);
  mutation.observe(root, {subtree: true, childList: true});
  window.__featurelens_mutation_observer = mutation;

  const restoreFocus = (block) => {
    if (!block || !block.classList.contains("featurelens-inline-focus")) return;
    const saved = block.__featurelens_saved_style;
    if (saved == null || saved === "") block.removeAttribute("style");
    else block.setAttribute("style", saved);
    block.classList.remove("featurelens-inline-focus");
    block.__featurelens_saved_style = null;
    window.setTimeout(kick, 30);
  };

  const closeOtherFocus = (except) => {
    document.querySelectorAll(".featurelens-inline-focus").forEach((node) => {
      if (node !== except) restoreFocus(node);
    });
  };

  const focusPlotInPlace = (block) => {
    const rect = block.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const viewportWidth = Math.max(320, document.documentElement.clientWidth || window.innerWidth || rect.width);
    const screenHeight = Math.max(600, (window.screen && window.screen.availHeight) || 900);
    const maxWidth = Math.min(viewportWidth * 0.90, 1100);
    const maxHeight = Math.min(screenHeight * 0.68, 700);
    const scale = Math.max(1, Math.min(maxWidth / rect.width, maxHeight / rect.height, 1.8));
    const focusedWidth = rect.width * scale;
    let dx = (viewportWidth - focusedWidth) / 2 - rect.left;
    if (rect.left + dx < 12) dx += 12 - (rect.left + dx);
    if (rect.left + dx + focusedWidth > viewportWidth - 12) {
      dx -= (rect.left + dx + focusedWidth) - (viewportWidth - 12);
    }
    block.style.transformOrigin = "top left";
    block.style.transform = `translate(${dx}px, 0px) scale(${scale})`;
    block.style.marginBottom = `${Math.max(8, rect.height * (scale - 1) + 8)}px`;
    block.style.zIndex = "5000";
  };

  const focusTableInPlace = (block) => {
    const rect = block.getBoundingClientRect();
    if (rect.width <= 0) return;
    const viewportWidth = Math.max(320, document.documentElement.clientWidth || window.innerWidth || rect.width);
    const targetWidth = Math.max(rect.width, Math.min(viewportWidth * 0.94, 1400));
    let dx = (viewportWidth - targetWidth) / 2 - rect.left;
    if (rect.left + dx < 12) dx += 12 - (rect.left + dx);
    if (rect.left + dx + targetWidth > viewportWidth - 12) {
      dx -= (rect.left + dx + targetWidth) - (viewportWidth - 12);
    }
    block.style.width = `${targetWidth}px`;
    block.style.maxWidth = "none";
    block.style.transform = `translateX(${dx}px)`;
    block.style.zIndex = "5000";
  };

  const toggleInlineFocus = (block) => {
    if (block.classList.contains("featurelens-inline-focus")) {
      restoreFocus(block);
      return;
    }
    closeOtherFocus(block);
    block.__featurelens_saved_style = block.getAttribute("style") || "";
    block.classList.add("featurelens-inline-focus");
    if (block.classList.contains("fl-plot")) focusPlotInPlace(block);
    else focusTableInPlace(block);
    window.setTimeout(kick, 30);
  };

  // Keep the native toolbar icon, but replace Gradio fullscreen with an in-place expansion.
  // This avoids HF iframe jumps and preserves the chart's exact rendered aspect ratio.
  document.addEventListener("click", (event) => {
    const button = event.target && event.target.closest ? event.target.closest("button") : null;
    if (!button) return;
    const label = `${button.getAttribute("aria-label") || ""} ${button.getAttribute("title") || ""} ${button.textContent || ""}`.toLowerCase();
    if (!label.includes("fullscreen")) return;
    const block = button.closest(".fl-plot, .result-table");
    if (!block) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    toggleInlineFocus(block);
  }, true);

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    const active = document.querySelector(".featurelens-inline-focus");
    if (active) restoreFocus(active);
  });

  // Rename Gradio's generic chart.png export without touching the export implementation.
  document.addEventListener("click", (event) => {
    const button = event.target && event.target.closest ? event.target.closest("button") : null;
    if (button) {
      const label = `${button.getAttribute("aria-label") || ""} ${button.getAttribute("title") || ""} ${button.textContent || ""}`.toLowerCase();
      if (label.includes("export")) {
        const block = button.closest(".fl-plot");
        if (block) {
          const id = block.id || "plot-featurelens-chart";
          const stem = id.replace(/^plot-/, "").replace(/[^a-z0-9_-]+/gi, "-").replace(/-+/g, "-").replace(/^-|-$/g, "");
          window.__featurelens_export_name = `featurelens_${stem || "chart"}.png`;
        }
      }
    }
    const anchor = event.target && event.target.closest ? event.target.closest('a[download="chart.png"]') : null;
    if (anchor && window.__featurelens_export_name) {
      anchor.setAttribute("download", window.__featurelens_export_name);
      window.setTimeout(() => { window.__featurelens_export_name = null; }, 500);
    }
  }, true);

  kick();
  return [];
}
"""


def _raise_ui_error(exc: Exception) -> None:
    raise gr.Error(f"{type(exc).__name__}: {exc}") from exc


def _tsv(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return ""
    return frame.to_csv(sep="\t", index=False, lineterminator="\n")


def _copy_button(label: str = "Copy TSV") -> gr.Button:
    return gr.Button(label, size="sm", variant="primary", elem_classes=["copy-btn"])


def _table_heading(text: str) -> gr.HTML:
    return gr.HTML(f'<div class="table-heading">{text}</div>')


def _copy_ack(_text: str) -> None:
    gr.Info("Copied TSV with headers.", duration=1.0)


def _bind_copy(button: gr.Button, source: gr.Textbox) -> None:
    button.click(fn=_copy_ack, inputs=[source], outputs=None, js=COPY_JS, queue=False)


def _analysis_metrics_markdown(result) -> str:
    return (
        "#### Analysis metrics\n"
        f"**Layer {result.layer} · prompt token {result.token_index}**  \n"
        f"Active SAE features: **{int(result.metrics['active_features'])}/{SETTINGS.sae_top_k}**  \n"
        f"Reconstruction cosine: **{result.metrics['cosine']:.4f}** · "
        f"NMSE: **{result.metrics['nmse']:.4f}**  \n"
        f"Top-5 activation mass: **{result.metrics['top5_mass_fraction']:.1%}**"
    )


def _intervention_metrics_markdown(result) -> str:
    drift = f"null drift JS **{result.execution_drift_js:.2e}**"
    if result.execution_drift_mean_logprob is not None:
        drift += f" · mean log p/token **{result.execution_drift_mean_logprob:+.2e}**"

    lines = [
        f"Feature activation **{result.feature_activation:.4f}** · Δ coefficient **{result.delta_activation:+.4f}** · perturbation L2 **{result.perturbation_norm:.4f}**",
        f"Next-token JS **{result.js_divergence:.6f}** · random mean **{result.random_js_divergence:.6f} ± {result.random_js_std:.6f}** · specificity **{result.js_specificity_ratio:.2f}×** · tail **{result.js_empirical_p:.3f}**",
        f"Execution context: {drift}",
    ]
    if result.baseline_sequence_logprob is not None:
        tokens = " ".join(repr(token) for token in result.target_tokens)
        lines.extend([
            f"Target {tokens} · baseline log p **{result.baseline_sequence_logprob:.4f}** · SAE edit **{result.modified_sequence_logprob:.4f}**",
            f"Δ mean log p/token **{result.mean_logprob_delta:+.4f}** · random mean |Δ| **{result.random_abs_mean_logprob_delta:.4f} ± {result.random_mean_logprob_std:.4f}** · specificity **{result.target_specificity_ratio:.2f}×** · tail **{result.target_empirical_p:.3f}**",
        ])
    if abs(result.feature_activation) < 1e-12:
        lines.append("_This feature is inactive at the selected token; ablate/scale therefore has zero native coefficient to remove._")
    elif result.baseline_text == result.modified_text:
        lines.append("_Greedy text is unchanged; the probability-level metrics above are more sensitive than deterministic decoding._")
    return "  \n".join(lines)

def _dose_metrics_markdown(result) -> str:
    tokens = " ".join(repr(token) for token in result.target_tokens)
    inactive = " · inactive at this token" if abs(result.feature_activation) < 1e-12 else ""
    return (
        f"Feature activation **{result.feature_activation:.4f}**{inactive} · target {tokens}  \n"
        f"Reference: **1× no edit** · execution drift mean log p/token **{result.execution_drift_mean_logprob:+.2e}** · JS **{result.execution_drift_js:.2e}**"
    )

def _feature_set_metrics_markdown(result) -> str:
    tokens = " ".join(repr(token) for token in result.target_tokens)
    inactive_count = sum(abs(float(row[1])) < 1e-12 for row in result.feature_rows)
    inactive_note = (
        f"  \n{inactive_count} selected feature(s) were inactive and contributed zero delta."
        if inactive_count
        else ""
    )
    return (
        f"Selected feature set: **{len(result.feature_ids)} features** · "
        f"perturbation L2: **{result.perturbation_norm:.4f}**  \n"
        f"Target continuation: {len(result.target_tokens)} token(s): {tokens}  \n"
        f"SAE Δ mean log p/token: **{result.mean_logprob_delta:+.4f}** · "
        f"random ensemble ({result.random_control_count}) mean |Δ|: "
        f"**{result.random_abs_mean_logprob_delta:.4f}** ± **{result.random_mean_logprob_std:.4f}** · ratio: **{result.target_specificity_ratio:.2f}×** · "
        f"empirical tail p: **{result.target_empirical_p:.3f}**  \n"
        f"SAE Δ sequence log p: **{result.sequence_logprob_delta:+.4f}** · "
        f"random signed mean Δ: **{result.random_sequence_logprob_delta:+.4f}**  \n"
        f"Next-token JS: **{result.js_divergence:.6f}** · random mean JS: "
        f"**{result.random_js_divergence:.6f}** ± **{result.random_js_std:.6f}** · "
        f"ratio: **{result.js_specificity_ratio:.2f}×** · empirical tail p: **{result.js_empirical_p:.3f}**  \n"
        f"Execution-context null drift: mean log p/token **{result.execution_drift_mean_logprob:+.2e}**, "
        f"JS **{result.execution_drift_js:.2e}**{inactive_note}"
    )


def _interaction_metrics_markdown(result) -> str:
    tokens = " ".join(repr(token) for token in result.target_tokens)
    return (
        f"Target {tokens} · additive expectation **{result.additive_expected_mean_delta:+.4f}** · "
        f"joint effect **{result.joint_mean_delta:+.4f}** · interaction excess **{result.interaction_excess_mean_delta:+.4f}** "
        f"(normalized **{result.normalized_interaction:+.3f}**)  \n"
        f"Execution drift **{result.execution_drift_mean_logprob:+.2e}** mean log p/token. "
        "_Non-additivity is downstream interaction evidence, not a circuit claim._"
    )

def _paraphrase_metrics_markdown(result) -> str:
    return (
        "**Selected token** · "
        f"Jaccard **{result.topk_jaccard:.3f}** · sparse cosine **{result.sparse_cosine:.3f}** · "
        f"shared displayed features **{result.shared_top_n}/{result.top_n}**  \n"
        "**Prompt-wide max pool** · "
        f"Jaccard **{result.promptwide_jaccard:.3f}** · cosine **{result.promptwide_cosine:.3f}**"
    )

def _concept_metrics_markdown(result) -> str:
    coverage = f"{result.active_prompt_count}/{result.total_prompt_count}"
    if result.leading_concept is None:
        leader = "inactive in every sampled prompt"
    elif result.leading_ratio is None:
        leader = f"highest mean: **{result.leading_concept}** (runner-up mean 0)"
    else:
        leader = f"highest mean: **{result.leading_concept}** (**{result.leading_ratio:.2f}×** runner-up)"
    return (
        f"Feature **{result.feature_id}** · layer **{result.layer}** · active in **{coverage}** prompts · {leader}.  \n"
        "_Exploratory prompt-wide contrast; use the Study tab for held-out evidence._"
    )

def _trace_metrics_markdown(result) -> str:
    if result.max_token_index is None:
        peak = "Feature is inactive at every prompt token."
    else:
        peak = (
            f"Peak activation **{result.max_activation:.4f}** at token **{result.max_token_index}** "
            f"({result.tokens[result.max_token_index]!r})."
        )
    return (
        f"Feature **{result.feature_id}**, layer **{result.layer}** · active at "
        f"**{result.active_token_count}/{result.token_count}** prompt tokens.  \n{peak}"
    )


def _geometry_metrics_markdown(result) -> str:
    if result.alignment_ratio > 1.05:
        geometry = "net aligned"
    elif result.alignment_ratio < 0.95:
        geometry = "net cancelling"
    else:
        geometry = "near the independent-direction reference"
    return (
        f"Features **{', '.join(str(x) for x in result.feature_ids)}** · layer **{result.layer}** · "
        f"mean |cos| **{result.mean_abs_decoder_cosine:.3f}** · max |cos| **{result.max_abs_decoder_cosine:.3f}**  \n"
        f"Joint L2 **{result.joint_ablation_norm:.4f}** · independent reference **{result.independent_norm:.4f}** · "
        f"ratio **{result.alignment_ratio:.3f}×** ({geometry})."
    )

def _contrastive_metrics_markdown(result) -> str:
    direction = "toward A" if result.delta_log_odds > 0 else ("toward B" if result.delta_log_odds < 0 else "no shift")
    return (
        f"Feature **{result.feature_id}** · activation **{result.feature_activation:.4f}** · perturbation L2 **{result.perturbation_norm:.4f}**  \n"
        f"Exact-sequence log-odds A−B: baseline **{result.baseline_log_odds:+.4f}** · edit **{result.modified_log_odds:+.4f}** · "
        f"shift **{result.delta_log_odds:+.4f}** ({direction})  \n"
        f"Token-normalized shift **{result.delta_normalized_preference:+.4f}** · random mean |Δ| **{result.random_abs_mean_delta:.4f} ± {result.random_delta_std:.4f}** · "
        f"specificity **{result.specificity_ratio:.2f}×** · tail **{result.empirical_p:.3f}**"
    )

def _discovery_metrics_markdown(result) -> str:
    if not result.candidate_ids:
        scope = "current-token-active " if result.ranking_mode == "causal_ready" else ""
        return f"No positively selective {scope}candidates found for **{result.concept}** at layer **{result.layer}** in this live batch."

    ranking = {
        "balanced_selectivity": "balanced selectivity",
        "raw_mean_difference": "raw mean difference",
        "causal_ready": "causal-ready evidence",
    }[result.ranking_mode]
    lines = [
        f"**{result.concept}** · layer **{result.layer}** · {result.prompts_per_concept} prompts/concept · **{len(result.candidate_ids)}** candidates by {ranking}",
    ]
    if result.current_context_available:
        lines.append(
            f"Current token **{result.current_token_index}** · active candidates **{result.displayed_current_active_count}/{len(result.candidate_ids)}**"
        )
    if result.split_half_jaccard is not None:
        lines.append(
            f"Split-half shortlist · shared **{result.split_half_shared_count}** · Jaccard **{result.split_half_jaccard:.3f}**"
        )
    if result.resample_replicates and result.resample_mean_support is not None:
        lines.append(
            f"{result.resample_replicates} resamples · mean shortlist support **{result.resample_mean_support:.1%}** · "
            f"≥75% support **{result.resample_high_support_count}/{len(result.candidate_ids)}** · not a confidence interval"
        )
    lines.append("_Live discovery is exploratory; semantic claims belong to the held-out Study results._")
    return "  \n".join(lines)

def _candidate_screen_metrics_markdown(result) -> str:
    tokens = " ".join(repr(token) for token in result.target_tokens)
    if result.rows:
        top = result.rows[0]
        strongest = f"top feature **{int(top[1])}** · Δ mean log p/token **{float(top[5]):+.4f}** · JS **{float(top[7]):.6f}**"
    else:
        strongest = "no candidate rows"
    return (
        f"Screened **{result.candidate_count}** candidates · active **{result.active_feature_count}** · target {tokens} · {strongest}  \n"
        f"Null drift mean log p/token **{result.execution_drift_mean_logprob:+.2e}** · JS **{result.execution_drift_js:.2e}**. "
        "_This is a triage screen; no random-control ensemble is spent here._"
    )

def _spearman_rank_corr(left: list[float], right: list[float]) -> float | None:
    """Descriptive Spearman correlation with tie-aware average ranks."""
    if len(left) != len(right) or len(left) < 2:
        return None
    left_s = pd.Series(left, dtype="float64")
    right_s = pd.Series(right, dtype="float64")
    if left_s.nunique(dropna=True) < 2 or right_s.nunique(dropna=True) < 2:
        return None
    value = left_s.rank(method="average").corr(right_s.rank(method="average"))
    return None if pd.isna(value) else float(value)


def _candidate_alignment_outputs(
    discovery_table: pd.DataFrame | None,
    screen_table: pd.DataFrame | None,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Join discovery evidence to causal triage results without another model call."""
    empty_columns = [
        "Feature id",
        "Discovery rank",
        "Target-effect rank",
        "Distribution-shift rank",
        "Candidate score",
        "Selectivity",
        "Current token activation",
        "|Δ mean log p/token|",
        "Next-token JS",
        "Discovery→target rank shift",
    ]
    if discovery_table is None or screen_table is None:
        return "", pd.DataFrame(columns=empty_columns), pd.DataFrame()
    discovery = pd.DataFrame(discovery_table).copy()
    screen = pd.DataFrame(screen_table).copy()
    if discovery.empty or screen.empty or "Feature id" not in discovery or "Feature id" not in screen:
        return "", pd.DataFrame(columns=empty_columns), pd.DataFrame()

    discovery["Feature id"] = pd.to_numeric(discovery["Feature id"], errors="coerce")
    screen["Feature id"] = pd.to_numeric(screen["Feature id"], errors="coerce")
    discovery = discovery.dropna(subset=["Feature id"]).copy()
    screen = screen.dropna(subset=["Feature id"]).copy()
    discovery["Feature id"] = discovery["Feature id"].astype(int)
    screen["Feature id"] = screen["Feature id"].astype(int)

    needed_discovery = {"Rank", "Candidate score", "Selectivity", "Current token activation"}
    needed_screen = {"Rank", "Δ mean log p/token", "Next-token JS"}
    if not needed_discovery.issubset(discovery.columns) or not needed_screen.issubset(screen.columns):
        return "", pd.DataFrame(columns=empty_columns), pd.DataFrame()

    discovery_lookup = discovery.set_index("Feature id", drop=False)
    js_ranked = screen.sort_values(["Next-token JS", "Feature id"], ascending=[False, True]).reset_index(drop=True)
    js_ranks = {int(row["Feature id"]): rank for rank, (_, row) in enumerate(js_ranked.iterrows(), start=1)}

    rows: list[list[object]] = []
    for _, causal_row in screen.iterrows():
        feature_id = int(causal_row["Feature id"])
        if feature_id not in discovery_lookup.index:
            continue
        discovery_row = discovery_lookup.loc[feature_id]
        # set_index can technically return a DataFrame for duplicate ids; use the first row deterministically.
        if isinstance(discovery_row, pd.DataFrame):
            discovery_row = discovery_row.iloc[0]
        discovery_rank = int(float(discovery_row["Rank"]))
        target_rank = int(float(causal_row["Rank"]))
        mean_delta = float(causal_row["Δ mean log p/token"])
        rows.append(
            [
                feature_id,
                discovery_rank,
                target_rank,
                int(js_ranks[feature_id]),
                float(discovery_row["Candidate score"]),
                float(discovery_row["Selectivity"]),
                float(discovery_row["Current token activation"]),
                abs(mean_delta),
                float(causal_row["Next-token JS"]),
                discovery_rank - target_rank,
            ]
        )

    table = pd.DataFrame(rows, columns=empty_columns)
    if table.empty:
        return "", table, pd.DataFrame()

    rho_target = _spearman_rank_corr(
        table["Candidate score"].astype(float).tolist(),
        table["|Δ mean log p/token|"].astype(float).tolist(),
    )
    rho_js = _spearman_rank_corr(
        table["Candidate score"].astype(float).tolist(),
        table["Next-token JS"].astype(float).tolist(),
    )

    top_discovery = table.sort_values(["Discovery rank", "Feature id"]).iloc[0]
    top_target = table.sort_values(["Target-effect rank", "Feature id"]).iloc[0]
    top_js = table.sort_values(["Distribution-shift rank", "Feature id"]).iloc[0]

    def fmt_rho(value: float | None) -> str:
        return "undefined" if value is None else f"{value:+.3f}"

    summary = (
        f"Compared **{len(table)}** screened candidates using the discovery evidence and causal triage from the same workflow.  \n"
        f"Top discovery candidate: **{int(top_discovery['Feature id'])}** · strongest target effect: "
        f"**{int(top_target['Feature id'])}** · strongest next-token distribution shift: **{int(top_js['Feature id'])}**.  \n"
        f"Spearman ρ(candidate score, |target effect|): **{fmt_rho(rho_target)}** · "
        f"ρ(candidate score, next-token JS): **{fmt_rho(rho_js)}**.  \n\n"
        "_descriptive shortlist comparison; triage does not use random controls._"
    )

    chart = table[["Feature id", "Candidate score", "|Δ mean log p/token|", "Discovery rank", "Target-effect rank", "Next-token JS"]].copy()
    chart["Feature id"] = chart["Feature id"].astype(str)
    chart["Series"] = "Screened candidate"
    return summary, table, chart



def _controlled_candidate_shortlist(
    discovery_table: pd.DataFrame | None,
    screen_table: pd.DataFrame | None,
    limit: int = 3,
) -> list[str]:
    """Pick a small controlled follow-up set without another model call.

    Prefer the strongest discovery candidate, strongest target-effect candidate,
    and strongest distribution-shift candidate. Fill any duplicate slots using
    target-effect rank. This preserves the association-vs-causality contrast
    instead of blindly testing only the triage top-k.
    """
    if screen_table is None:
        return []
    screen = pd.DataFrame(screen_table).copy()
    if screen.empty or "Feature id" not in screen:
        return []
    screen["Feature id"] = pd.to_numeric(screen["Feature id"], errors="coerce")
    screen = screen.dropna(subset=["Feature id"]).copy()
    screen["Feature id"] = screen["Feature id"].astype(int)
    if screen.empty:
        return []

    selected: list[int] = []

    def add(feature_id: int) -> None:
        if feature_id not in selected and len(selected) < int(limit):
            selected.append(feature_id)

    if discovery_table is not None:
        discovery = pd.DataFrame(discovery_table).copy()
        if not discovery.empty and {"Feature id", "Rank"}.issubset(discovery.columns):
            discovery["Feature id"] = pd.to_numeric(discovery["Feature id"], errors="coerce")
            discovery = discovery.dropna(subset=["Feature id"]).copy()
            discovery["Feature id"] = discovery["Feature id"].astype(int)
            screened_ids = set(screen["Feature id"].astype(int).tolist())
            discovery = discovery[discovery["Feature id"].isin(screened_ids)]
            if not discovery.empty:
                top_discovery = discovery.sort_values(["Rank", "Feature id"]).iloc[0]
                add(int(top_discovery["Feature id"]))

    target_sorted = screen.sort_values(["Rank", "Feature id"])
    add(int(target_sorted.iloc[0]["Feature id"]))

    if "Next-token JS" in screen.columns:
        js_sorted = screen.sort_values(["Next-token JS", "Feature id"], ascending=[False, True])
        add(int(js_sorted.iloc[0]["Feature id"]))

    for feature_id in target_sorted["Feature id"].astype(int).tolist():
        add(feature_id)
        if len(selected) >= int(limit):
            break
    return [str(feature_id) for feature_id in selected]


def _candidate_specificity_metrics_markdown(result) -> str:
    tokens = " ".join(repr(token) for token in result.target_tokens)
    strongest = "no controlled rows"
    if result.rows:
        top = result.rows[0]
        strongest = f"top target specificity **{int(top[1])}** at **{float(top[9]):.2f}×** random mean |effect| · tail **{float(top[10]):.3f}**"
    return (
        f"Compared **{result.candidate_count}** candidates · active **{result.active_feature_count}** · "
        f"**{result.random_control_count}** random controls each · target {tokens} · {strongest}  \n"
        f"Null drift mean log p/token **{result.execution_drift_mean_logprob:+.2e}** · JS **{result.execution_drift_js:.2e}**"
    )

def _controlled_evidence_patterns(
    specificity_table: pd.DataFrame | None,
) -> tuple[str, pd.DataFrame]:
    """Summarize controlled target-vs-distribution specificity without requiring discovery state."""
    columns = [
        "Feature id",
        "Target specificity ratio",
        "JS specificity ratio",
        "Target empirical tail p",
        "JS empirical tail p",
        "Evidence pattern",
        "Interpretation",
    ]
    if specificity_table is None:
        return "", pd.DataFrame(columns=columns)
    table = pd.DataFrame(specificity_table).copy()
    needed = {
        "Feature id",
        "Target specificity ratio",
        "JS specificity ratio",
        "Target empirical tail p",
        "JS empirical tail p",
    }
    if table.empty or not needed.issubset(table.columns):
        return "", pd.DataFrame(columns=columns)

    rows: list[list[object]] = []
    for _, row in table.iterrows():
        feature_id = int(float(row["Feature id"]))
        target_ratio = float(row["Target specificity ratio"])
        js_ratio = float(row["JS specificity ratio"])
        target_p = float(row["Target empirical tail p"])
        js_p = float(row["JS empirical tail p"])
        if target_ratio >= 1.5 and js_ratio >= 1.5:
            pattern = "Broad controlled influence"
            interpretation = "Both the specified target and the local next-token distribution exceed matched-random magnitude baselines."
        elif target_ratio >= 1.5:
            pattern = "Target-weighted"
            interpretation = "The specified continuation is affected more strongly than the broader distributional diagnostic."
        elif js_ratio >= 1.5 and target_ratio < 1.0:
            pattern = "Distribution-shift dominant"
            interpretation = "The feature changes the local distribution beyond matched random directions without selectively controlling this target."
        elif js_ratio >= 1.5:
            pattern = "Distribution-shift weighted"
            interpretation = "Distributional influence is clearer than target-specific influence for the tested continuation."
        else:
            pattern = "Weak / mixed specificity"
            interpretation = "Neither live specificity ratio clearly dominates its matched-random baseline."
        rows.append([feature_id, target_ratio, js_ratio, target_p, js_p, pattern, interpretation])

    out = pd.DataFrame(rows, columns=columns)
    descriptions = "; ".join(
        f"**{int(row['Feature id'])}**: {row['Evidence pattern']}" for _, row in out.iterrows()
    )
    summary = (
        f"Controlled evidence patterns — {descriptions}.  \n\n"
        "_Pattern labels summarize effect ratios, not statistical significance._"
    )
    return summary, out


def _cross_target_shortlist(specificity_table: pd.DataFrame | None, limit: int = 2) -> list[str]:
    if specificity_table is None:
        return []
    table = pd.DataFrame(specificity_table).copy()
    needed = {"Feature id", "Target specificity ratio", "JS specificity ratio"}
    if table.empty or not needed.issubset(table.columns):
        return []
    table["Feature id"] = pd.to_numeric(table["Feature id"], errors="coerce")
    table = table.dropna(subset=["Feature id"]).copy()
    table["Feature id"] = table["Feature id"].astype(int)
    selected: list[int] = []

    def add(feature_id: int) -> None:
        if feature_id not in selected and len(selected) < int(limit):
            selected.append(feature_id)

    target = table.sort_values(["Target specificity ratio", "Feature id"], ascending=[False, True])
    js = table.sort_values(["JS specificity ratio", "Feature id"], ascending=[False, True])
    if not target.empty:
        add(int(target.iloc[0]["Feature id"]))
    if not js.empty:
        add(int(js.iloc[0]["Feature id"]))
    for feature_id in target["Feature id"].tolist():
        add(int(feature_id))
    return [str(feature_id) for feature_id in selected]


def _cross_target_metrics_markdown(result) -> str:
    feature_text = ", ".join(str(feature_id) for feature_id in result.feature_ids)
    target_text = ", ".join(repr(target) for target in result.targets)
    series_text = " · ".join(
        f"Feature {chr(65 + idx)} = **{feature_id}**" for idx, feature_id in enumerate(result.feature_ids)
    )
    lines = [
        f"Features **{feature_text}** · targets {target_text} · active **{result.active_feature_count}/{len(result.feature_ids)}**",
        series_text,
    ]
    if result.summary_rows:
        strongest = result.summary_rows[0]
        lines.append(
            f"Largest effect: **{int(strongest[0])}** on **{strongest[1]!r}** · Δ mean log p/token **{float(strongest[2]):+.4f}**"
        )
        patterns = "; ".join(f"{int(row[0])}: {row[10]}" for row in result.summary_rows)
        lines.append(f"Profiles · {patterns}")
    if result.pairwise_rows:
        top_pair = result.pairwise_rows[0]
        lines.append(
            f"Largest pairwise preference shift: **{int(top_pair[0])}** · {top_pair[1]!r} vs {top_pair[2]!r} · **{float(top_pair[3]):+.4f}**"
        )
    lines.append("_Target-profile screen; matched-random specificity is reported above._")
    return "  \n".join(lines)

def _controlled_alignment_outputs(
    discovery_table: pd.DataFrame | None,
    specificity_table: pd.DataFrame | None,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Join concept evidence to random-controlled causal specificity."""
    columns = [
        "Feature id",
        "Discovery rank",
        "Specificity rank",
        "Target-effect rank",
        "JS-specificity rank",
        "Candidate score",
        "|SAE Δ mean log p/token|",
        "Random mean |Δ|",
        "Target specificity ratio",
        "Target empirical tail p",
        "SAE next-token JS",
        "Random mean JS",
        "JS specificity ratio",
        "JS empirical tail p",
        "Discovery→specificity rank shift",
    ]
    if specificity_table is None:
        return "", pd.DataFrame(columns=columns), pd.DataFrame()
    if discovery_table is None:
        return (
            "Controlled results are available, but discovery evidence is not present in this browser session, so rank alignment cannot be reconstructed.",
            pd.DataFrame(columns=columns),
            pd.DataFrame(),
        )
    discovery = pd.DataFrame(discovery_table).copy()
    controlled = pd.DataFrame(specificity_table).copy()
    if controlled.empty or "Feature id" not in controlled:
        return "", pd.DataFrame(columns=columns), pd.DataFrame()
    if discovery.empty or "Feature id" not in discovery:
        return (
            "Controlled results are available, but discovery evidence is not currently populated.",
            pd.DataFrame(columns=columns),
            pd.DataFrame(),
        )

    discovery["Feature id"] = pd.to_numeric(discovery["Feature id"], errors="coerce")
    controlled["Feature id"] = pd.to_numeric(controlled["Feature id"], errors="coerce")
    discovery = discovery.dropna(subset=["Feature id"]).copy()
    controlled = controlled.dropna(subset=["Feature id"]).copy()
    discovery["Feature id"] = discovery["Feature id"].astype(int)
    controlled["Feature id"] = controlled["Feature id"].astype(int)

    needed_discovery = {"Rank", "Candidate score"}
    needed_controlled = {
        "Rank",
        "SAE Δ mean log p/token",
        "Random mean |Δ|",
        "Target specificity ratio",
        "Target empirical tail p",
        "SAE next-token JS",
        "Random mean JS",
        "JS specificity ratio",
        "JS empirical tail p",
    }
    if not needed_discovery.issubset(discovery.columns) or not needed_controlled.issubset(controlled.columns):
        return "", pd.DataFrame(columns=columns), pd.DataFrame()

    discovery_lookup = discovery.set_index("Feature id", drop=False)
    target_ranked = controlled.assign(
        _abs_target=controlled["SAE Δ mean log p/token"].astype(float).abs()
    ).sort_values(["_abs_target", "Feature id"], ascending=[False, True])
    target_ranks = {
        int(row["Feature id"]): rank for rank, (_, row) in enumerate(target_ranked.iterrows(), start=1)
    }
    js_ranked = controlled.sort_values(
        ["JS specificity ratio", "Feature id"], ascending=[False, True]
    )
    js_ranks = {
        int(row["Feature id"]): rank for rank, (_, row) in enumerate(js_ranked.iterrows(), start=1)
    }

    rows: list[list[object]] = []
    for _, row in controlled.iterrows():
        feature_id = int(row["Feature id"])
        if feature_id not in discovery_lookup.index:
            continue
        discovery_row = discovery_lookup.loc[feature_id]
        if isinstance(discovery_row, pd.DataFrame):
            discovery_row = discovery_row.iloc[0]
        discovery_rank = int(float(discovery_row["Rank"]))
        specificity_rank = int(float(row["Rank"]))
        rows.append(
            [
                feature_id,
                discovery_rank,
                specificity_rank,
                int(target_ranks[feature_id]),
                int(js_ranks[feature_id]),
                float(discovery_row["Candidate score"]),
                abs(float(row["SAE Δ mean log p/token"])),
                float(row["Random mean |Δ|"]),
                float(row["Target specificity ratio"]),
                float(row["Target empirical tail p"]),
                float(row["SAE next-token JS"]),
                float(row["Random mean JS"]),
                float(row["JS specificity ratio"]),
                float(row["JS empirical tail p"]),
                discovery_rank - specificity_rank,
            ]
        )

    table = pd.DataFrame(rows, columns=columns)
    if table.empty:
        return "", table, pd.DataFrame()

    rho_target = _spearman_rank_corr(
        table["Candidate score"].astype(float).tolist(),
        table["Target specificity ratio"].astype(float).tolist(),
    )
    rho_js = _spearman_rank_corr(
        table["Candidate score"].astype(float).tolist(),
        table["JS specificity ratio"].astype(float).tolist(),
    )
    top_discovery = table.sort_values(["Discovery rank", "Feature id"]).iloc[0]
    top_specificity = table.sort_values(["Specificity rank", "Feature id"]).iloc[0]
    top_js = table.sort_values(["JS-specificity rank", "Feature id"]).iloc[0]

    def fmt_rho(value: float | None) -> str:
        return "undefined" if value is None else f"{value:+.3f}"

    summary = (
        f"Controlled comparison covers **{len(table)}** candidates from the discovery/triage workflow.  \n"
        f"Top discovery candidate: **{int(top_discovery['Feature id'])}** · strongest random-normalized target effect: "
        f"**{int(top_specificity['Feature id'])}** · strongest random-normalized JS shift: **{int(top_js['Feature id'])}**.  \n"
        f"Spearman ρ(candidate score, target-specificity ratio): **{fmt_rho(rho_target)}** · "
        f"ρ(candidate score, JS-specificity ratio): **{fmt_rho(rho_js)}**.  \n\n"
        "This is stronger than the cheap triage comparison because each candidate is normalized against its own "
        "norm-matched random ensemble. The candidate count and eight-control ensemble are still small, so the correlations "
        "and empirical tails are descriptive live diagnostics rather than significance claims."
    )

    chart = table[
        ["Feature id", "Candidate score", "Target specificity ratio", "Discovery rank", "Specificity rank", "Target empirical tail p"]
    ].copy()
    chart["Feature id"] = chart["Feature id"].astype(str)
    chart["Series"] = "Controlled candidate"
    return summary, table, chart


def _cue_context_metrics_markdown(result) -> str:
    active = " · ".join(
        f"{cue} {count}/{len(result.stems)}" for cue, count in result.cue_active_context_counts.items()
    )
    if result.dominant_cue is None or result.active_condition_count == 0:
        interpretation = "No tested cue activated the feature."
    elif result.dominant_cue_context_count == len(result.stems) and result.off_dominant_active_count == 0:
        interpretation = f"**Cue-dominant pattern:** `{result.dominant_cue}` activates in every tested context; this is more consistent with a lexical/cue-specific response in this matrix."
    else:
        interpretation = f"Strongest cue: `{result.dominant_cue}` ({result.dominant_cue_context_count}/{len(result.stems)} contexts); pattern remains context-dependent."
    return (
        f"Feature **{result.feature_id}** · layer **{result.layer}** · active **{result.active_condition_count}/{result.condition_count}** conditions  \n"
        f"Cue coverage · {active}  \n{interpretation}"
    )

def _cue_metrics_markdown(result) -> str:
    return (
        f"Feature **{result.feature_id}** · layer **{result.layer}** · active for "
        f"**{result.active_cue_count}/{result.cue_count}** tested cues at the final token."
    )

def _global_context_markdown(prompt: str, layer: int, result) -> str:
    token = result.tokens[result.token_index] if result.tokens else ""
    short = prompt[:120] + ("…" if len(prompt) > 120 else "")
    return (
        f"**Context** · layer **{int(layer)}** · token **{result.token_index}** ({token!r}) · `{short}`"
    )

@gpu(duration=30)
def analyze_prompt(prompt: str, layer: int, token_index: int, top_n: int):
    try:
        if not prompt.strip():
            raise ValueError("Enter a prompt first.")
        result = RUNTIME.analyze(prompt, int(layer), int(token_index), int(top_n))
        columns = ["Rank", "Feature id", "Activation", "Offline concept hint"]
        table = pd.DataFrame(result.rows, columns=columns)
        choices = [str(int(row[1])) for row in result.rows]
        feature_update = gr.update(choices=choices, value=choices[0] if choices else None)
        feature_set_update = gr.update(choices=choices, value=choices[: min(3, len(choices))])
        contrast_update = gr.update(choices=choices, value=choices[0] if choices else None)
        chart_df = pd.DataFrame(
            {
                "Feature": [str(int(row[1])) for row in result.rows],
                "Activation": [float(row[2]) for row in result.rows],
                "Series": ["Activation"] * len(result.rows),
            }
        )
        location = (
            f"Current Workbench location — **layer {int(layer)}**, **prompt token {result.token_index}**; "
            f"prompt: `{prompt[:90]}{'…' if len(prompt) > 90 else ''}`"
        )
        return (
            RUNTIME.token_html(result.tokens, result.token_index),
            table,
            chart_df,
            feature_update,
            gr.update(choices=choices, value=choices[0] if choices else None),
            gr.update(choices=choices, value=choices[0] if choices else None),
            feature_set_update,
            contrast_update,
            gr.update(value=int(layer)),
            _analysis_metrics_markdown(result),
            location,
            location,
            _global_context_markdown(prompt, int(layer), result),
            _tsv(table),
        )
    except Exception as exc:
        _raise_ui_error(exc)


@gpu(duration=45)
def run_intervention(
    prompt: str,
    layer: int,
    token_index: int,
    feature_id: str,
    mode: str,
    coefficient: float,
    target_text: str,
    max_new_tokens: int,
):
    try:
        if not prompt.strip():
            raise ValueError("Enter a prompt first.")
        if feature_id is None or str(feature_id).strip() == "":
            raise ValueError("Choose or enter a feature id.")
        result = RUNTIME.intervene(
            text=prompt,
            layer=int(layer),
            token_index=int(token_index),
            feature_id=int(float(feature_id)),
            mode=mode,
            coefficient=float(coefficient),
            target_text=target_text,
            max_new_tokens=int(max_new_tokens),
        )
        token_columns = ["Token", "Baseline p", "SAE-edit p", "Δ probability"]
        target_columns = [
            "Target position",
            "Target token",
            "Baseline log p",
            "SAE-edit log p",
            "Random-ensemble mean log p",
            "SAE Δ log p",
            "Random-ensemble mean Δ log p",
        ]
        token_df = pd.DataFrame(result.top_token_rows, columns=token_columns)
        target_df = pd.DataFrame(result.target_token_rows, columns=target_columns)
        return (
            result.baseline_text,
            result.modified_text,
            _intervention_metrics_markdown(result),
            token_df,
            target_df,
            _tsv(token_df),
            _tsv(target_df),
        )
    except Exception as exc:
        _raise_ui_error(exc)


@gpu(duration=35)
def run_dose_response(prompt: str, layer: int, token_index: int, feature_id: str, target_text: str):
    try:
        if not prompt.strip():
            raise ValueError("Enter a prompt first.")
        if feature_id is None or str(feature_id).strip() == "":
            raise ValueError("Choose or enter a feature id.")
        if not target_text.strip():
            raise ValueError("Enter a target continuation before running the scale dose-response.")
        result = RUNTIME.dose_response(
            text=prompt,
            layer=int(layer),
            token_index=int(token_index),
            feature_id=int(float(feature_id)),
            target_text=target_text,
        )
        columns = [
            "Multiplier",
            "Δ feature coefficient",
            "Perturbation L2",
            "Batched-null mean log p/token",
            "Modified mean log p/token",
            "Δ mean log p/token",
            "Δ sequence log p",
            "Next-token JS",
        ]
        table = pd.DataFrame(result.rows, columns=columns)
        plot = table[["Multiplier", "Δ mean log p/token"]].copy()
        plot["Series"] = "SAE feature"
        return table, plot, _dose_metrics_markdown(result), _tsv(table)
    except Exception as exc:
        _raise_ui_error(exc)


@gpu(duration=35)
def run_layer_sweep(prompt: str, token_index: int):
    try:
        if not prompt.strip():
            raise ValueError("Enter a prompt first.")
        result = RUNTIME.layer_sweep(prompt, int(token_index))
        columns = [
            "Layer",
            "Reconstruction cosine",
            "NMSE",
            "Active features",
            "Top activation",
            "Top-5 mass",
            "Activation entropy",
        ]
        table = pd.DataFrame(result.rows, columns=columns)
        long = table.melt(
            id_vars=["Layer"],
            value_vars=["Reconstruction cosine", "Top-5 mass", "Activation entropy"],
            var_name="Metric",
            value_name="Value",
        )
        return RUNTIME.token_html(result.tokens, result.token_index), table, long, _tsv(table)
    except Exception as exc:
        _raise_ui_error(exc)


@gpu(duration=40)
def run_feature_set(
    prompt: str,
    layer: int,
    token_index: int,
    feature_ids: list[str] | None,
    mode: str,
    coefficient: float,
    target_text: str,
):
    try:
        if not prompt.strip():
            raise ValueError("Enter and inspect a prompt in the Workbench first.")
        selected = [int(float(value)) for value in (feature_ids or [])]
        if not selected:
            raise ValueError("Select at least one feature in 'Feature set'.")
        if not target_text.strip():
            raise ValueError("Enter a target continuation for the feature-set causal test.")
        result = RUNTIME.intervene_feature_set(
            text=prompt,
            layer=int(layer),
            token_index=int(token_index),
            feature_ids=selected,
            mode=mode,
            coefficient=float(coefficient),
            target_text=target_text,
        )
        feature_columns = ["Feature id", "Original activation", "Δ coefficient", "Offline concept hint"]
        target_columns = [
            "Target position",
            "Target token",
            "Baseline log p",
            "SAE-edit log p",
            "Random-ensemble mean log p",
            "SAE Δ log p",
            "Random-ensemble mean Δ log p",
        ]
        feature_df = pd.DataFrame(result.feature_rows, columns=feature_columns)
        target_df = pd.DataFrame(result.target_token_rows, columns=target_columns)
        return (
            feature_df,
            _feature_set_metrics_markdown(result),
            target_df,
            _tsv(feature_df),
            _tsv(target_df),
        )
    except Exception as exc:
        _raise_ui_error(exc)


@gpu(duration=45)
def run_feature_set_sweep(prompt: str, layer: int, token_index: int, target_text: str):
    try:
        if not prompt.strip():
            raise ValueError("Enter and inspect a prompt in the Workbench first.")
        if not target_text.strip():
            raise ValueError("Enter a target continuation before running the set-size sweep.")
        result = RUNTIME.feature_set_size_sweep(
            text=prompt,
            layer=int(layer),
            token_index=int(token_index),
            target_text=target_text,
        )
        columns = [
            "Set size k",
            "Feature ids",
            "Perturbation L2",
            "Batched-null mean log p/token",
            "SAE mean log p/token",
            "SAE Δ mean log p/token",
            "Random signed mean Δ",
            "Random mean |Δ|",
            "Random |Δ| std",
            "SAE/random magnitude ratio",
            "Empirical tail p",
            "SAE Δ sequence log p",
            "SAE next-token JS",
            "Random mean JS",
            "Random JS std",
            "JS empirical tail p",
        ]
        table = pd.DataFrame(result.rows, columns=columns)
        plot_rows: list[list[object]] = []
        for _, row in table.iterrows():
            plot_rows.append([row["Set size k"], "Top-k SAE ablation", row["SAE Δ mean log p/token"]])
            plot_rows.append([row["Set size k"], "Random signed mean", row["Random signed mean Δ"]])
        plot = pd.DataFrame(plot_rows, columns=["Set size k", "Condition", "Δ mean log p/token"])
        tokens = " ".join(repr(token) for token in result.target_tokens)
        note = (
            f"Target continuation: {len(result.target_tokens)} token(s): {tokens}. For each k, FeatureLens "
            f"ablates the k strongest active features and compares the effect with **{result.random_control_count} "
            f"norm-matched random directions**. All conditions share one batched zero-edit reference.  \n"
            f"Execution-context null drift: mean log p/token **{result.execution_drift_mean_logprob:+.2e}**, "
            f"JS **{result.execution_drift_js:.2e}**. The live empirical p-value is intentionally coarse because "
            f"it uses only {result.random_control_count} controls; the offline experiment should use more."
        )
        return table, plot, note, _tsv(table)
    except Exception as exc:
        _raise_ui_error(exc)


@gpu(duration=40)
def run_feature_interaction(
    prompt: str,
    layer: int,
    token_index: int,
    feature_ids: list[str] | None,
    target_text: str,
):
    try:
        selected = [int(float(value)) for value in (feature_ids or [])]
        if not prompt.strip():
            raise ValueError("Enter and inspect a prompt in the Workbench first.")
        if len(selected) < 2:
            raise ValueError("Select at least two features in 'Feature set'.")
        if len(selected) > 5:
            raise ValueError("Select at most five features for the interaction decomposition.")
        if not target_text.strip():
            raise ValueError("Enter a target continuation for the interaction decomposition.")
        result = RUNTIME.feature_interaction_test(
            text=prompt,
            layer=int(layer),
            token_index=int(token_index),
            feature_ids=selected,
            target_text=target_text,
        )
        columns = [
            "Condition",
            "Feature ids",
            "Activation summary",
            "Perturbation L2",
            "Δ mean log p/token",
            "Δ sequence log p",
            "Next-token JS",
        ]
        table = pd.DataFrame(result.rows, columns=columns)
        plot = table[["Condition", "Δ mean log p/token"]].copy()
        plot["Series"] = "Ablation effect"
        return table, _interaction_metrics_markdown(result), plot, _tsv(table)
    except Exception as exc:
        _raise_ui_error(exc)


@gpu(duration=30)
def run_paraphrase_compare(
    original_prompt: str,
    paraphrase_prompt: str,
    layer: int,
    token_index_a: int,
    token_index_b: int,
    top_n: int,
):
    try:
        result = RUNTIME.compare_paraphrases(
            text_a=original_prompt,
            text_b=paraphrase_prompt,
            layer=int(layer),
            token_index_a=int(token_index_a),
            token_index_b=int(token_index_b),
            top_n=int(top_n),
        )
        columns = ["Feature id", "Original activation", "Paraphrase activation", "Status", "Offline concept hint"]
        table = pd.DataFrame(result.rows, columns=columns)
        chart = pd.DataFrame(result.chart_rows, columns=["Feature", "Prompt", "Activation"])
        return (
            RUNTIME.token_html(result.tokens_a, result.token_index_a),
            RUNTIME.token_html(result.tokens_b, result.token_index_b),
            _paraphrase_metrics_markdown(result),
            table,
            chart,
            _tsv(table),
        )
    except Exception as exc:
        _raise_ui_error(exc)


@gpu(duration=35)
def run_concept_contrast(feature_id: str, layer: int, prompts_per_concept: int):
    try:
        if feature_id is None or str(feature_id).strip() == "":
            raise ValueError("Choose a feature id first. Run Workbench inspection if the selector is empty.")
        result = RUNTIME.concept_contrast_scan(
            feature_id=int(float(feature_id)),
            layer=int(layer),
            prompts_per_concept=int(prompts_per_concept),
        )
        columns = [
            "Concept",
            "Prompts",
            "Mean prompt-wide max",
            "Median prompt-wide max",
            "Prompt activation rate",
            "Mean when active",
            "Max activation",
        ]
        table = pd.DataFrame(result.rows, columns=columns)
        chart = pd.DataFrame(result.chart_rows, columns=["Concept", "Mean prompt-wide max"])
        chart["Series"] = "Prompt-wide max"
        return _concept_metrics_markdown(result), table, chart, _tsv(table)
    except Exception as exc:
        _raise_ui_error(exc)


@gpu(duration=30)
def run_feature_trace(prompt: str, layer: int, feature_id: str):
    try:
        if not prompt.strip():
            raise ValueError("Enter a prompt first.")
        if feature_id is None or str(feature_id).strip() == "":
            raise ValueError("Choose a feature id first.")
        result = RUNTIME.feature_token_trace(
            text=prompt,
            layer=int(layer),
            feature_id=int(float(feature_id)),
        )
        columns = ["Token position", "Token", "Activation", "Active in TopK"]
        table = pd.DataFrame(result.rows, columns=columns)
        chart = pd.DataFrame(result.chart_rows, columns=["Token", "Activation"])
        chart["Series"] = "Feature activation"
        return _trace_metrics_markdown(result), table, chart, _tsv(table)
    except Exception as exc:
        _raise_ui_error(exc)


@gpu(duration=30)
def run_feature_geometry(prompt: str, layer: int, token_index: int, feature_ids: list[str] | None):
    try:
        selected = [int(float(value)) for value in (feature_ids or [])]
        result = RUNTIME.feature_geometry(
            text=prompt,
            layer=int(layer),
            token_index=int(token_index),
            feature_ids=selected,
        )
        columns = ["Feature A", "Feature B", "Activation A", "Activation B", "Decoder cosine"]
        table = pd.DataFrame(result.rows, columns=columns)
        chart = pd.DataFrame(result.chart_rows, columns=["Feature pair", "Decoder cosine"])
        chart["Series"] = "Decoder cosine"
        return _geometry_metrics_markdown(result), table, chart, _tsv(table)
    except Exception as exc:
        _raise_ui_error(exc)


@gpu(duration=40)
def run_contrastive_causal(
    prompt: str,
    layer: int,
    token_index: int,
    feature_id: str,
    mode: str,
    coefficient: float,
    target_a: str,
    target_b: str,
):
    try:
        if feature_id is None or str(feature_id).strip() == "":
            raise ValueError("Choose a feature id first.")
        result = RUNTIME.contrastive_intervention(
            text=prompt,
            layer=int(layer),
            token_index=int(token_index),
            feature_id=int(float(feature_id)),
            mode=mode,
            coefficient=float(coefficient),
            target_a=target_a,
            target_b=target_b,
        )
        columns = [
            "Continuation",
            "Text",
            "Tokens",
            "Baseline sequence log p",
            "SAE-edit sequence log p",
            "Δ sequence log p",
            "Baseline mean log p/token",
            "SAE-edit mean log p/token",
            "Δ mean log p/token",
        ]
        table = pd.DataFrame(result.rows, columns=columns)
        chart = pd.DataFrame(
            [
                ["Baseline", result.baseline_log_odds],
                ["SAE edit", result.modified_log_odds],
            ],
            columns=["Condition", "A−B sequence log-odds"],
        )
        chart["Series"] = "Contrastive preference"
        return _contrastive_metrics_markdown(result), table, chart, _tsv(table)
    except Exception as exc:
        _raise_ui_error(exc)



@gpu(duration=35)
def run_concept_feature_discovery(
    concept: str,
    layer: int,
    prompts_per_concept: int,
    top_n: int,
    ranking_label: str,
    workbench_prompt: str,
    workbench_token_index: int,
):
    try:
        ranking_mode = {
            "Balanced selectivity": "balanced_selectivity",
            "Raw mean difference": "raw_mean_difference",
            "Causal-ready at current token": "causal_ready",
        }[ranking_label]
        result = RUNTIME.concept_feature_discovery(
            concept=concept,
            layer=int(layer),
            prompts_per_concept=int(prompts_per_concept),
            top_n=int(top_n),
            ranking_mode=ranking_mode,
            current_text=workbench_prompt,
            current_token_index=int(workbench_token_index),
        )
        columns = [
            "Rank",
            "Feature id",
            "Candidate score",
            "Target mean max",
            "Other mean max",
            "Mean difference",
            "Selectivity",
            "Target activation rate",
            "Other activation rate",
            "Current prompt max",
            "Current token activation",
            "Active at current token",
            "Resample shortlist support",
            "Median resample rank",
        ]
        table = pd.DataFrame(result.rows, columns=columns)
        chart = pd.DataFrame(result.chart_rows, columns=["Feature", "Candidate score"])
        chart["Series"] = "Candidate score"
        choices = [str(fid) for fid in result.candidate_ids]
        default = str(result.default_candidate_id) if result.default_candidate_id is not None else (choices[0] if choices else None)
        candidate_update = gr.update(choices=choices, value=default)
        screen_update = gr.update(
            choices=choices,
            value=choices[: min(5, len(choices))],
        )
        return (
            _discovery_metrics_markdown(result),
            table,
            chart,
            candidate_update,
            screen_update,
            _tsv(table),
        )
    except Exception as exc:
        _raise_ui_error(exc)



@gpu(duration=30)
def run_candidate_causal_screen(
    prompt: str,
    layer: int,
    token_index: int,
    feature_ids: list[str] | None,
    target_text: str,
    discovery_table: pd.DataFrame | None,
):
    try:
        selected = [int(float(value)) for value in (feature_ids or [])]
        result = RUNTIME.candidate_causal_screen(
            text=prompt,
            layer=int(layer),
            token_index=int(token_index),
            feature_ids=selected,
            target_text=target_text,
        )
        columns = [
            "Rank",
            "Feature id",
            "Native activation",
            "Active at current token",
            "Perturbation L2",
            "Δ mean log p/token",
            "Δ sequence log p",
            "Next-token JS",
        ]
        table = pd.DataFrame(result.rows, columns=columns)
        chart = pd.DataFrame(
            result.chart_rows,
            columns=["Feature", "Δ mean log p/token"],
        )
        chart["Series"] = "Candidate ablation"
        choices = [str(feature_id) for feature_id in result.feature_ids]
        candidate_update = gr.update(
            choices=choices,
            value=choices[0] if choices else None,
        )
        alignment_metrics, alignment_table, alignment_chart = _candidate_alignment_outputs(
            discovery_table, table
        )
        specificity_shortlist = _controlled_candidate_shortlist(discovery_table, table, limit=3)
        specificity_update = gr.update(
            choices=choices,
            value=specificity_shortlist,
        )
        return (
            _candidate_screen_metrics_markdown(result),
            table,
            chart,
            candidate_update,
            _tsv(table),
            alignment_metrics,
            alignment_table,
            alignment_chart,
            _tsv(alignment_table),
            specificity_update,
        )
    except Exception as exc:
        _raise_ui_error(exc)


@gpu(duration=40)
def run_candidate_specificity_screen(
    prompt: str,
    layer: int,
    token_index: int,
    feature_ids: list[str] | None,
    target_text: str,
    discovery_table: pd.DataFrame | None,
):
    try:
        selected = [int(float(value)) for value in (feature_ids or [])]
        result = RUNTIME.candidate_specificity_screen(
            text=prompt,
            layer=int(layer),
            token_index=int(token_index),
            feature_ids=selected,
            target_text=target_text,
        )
        columns = [
            "Rank",
            "Feature id",
            "Native activation",
            "Active at current token",
            "Perturbation L2",
            "SAE Δ mean log p/token",
            "Random signed mean Δ",
            "Random mean |Δ|",
            "Random |Δ| std",
            "Target specificity ratio",
            "Target empirical tail p",
            "SAE Δ sequence log p",
            "SAE next-token JS",
            "Random mean JS",
            "Random JS std",
            "JS specificity ratio",
            "JS empirical tail p",
        ]
        table = pd.DataFrame(result.rows, columns=columns)
        chart = pd.DataFrame(
            result.chart_rows,
            columns=["Feature", "Specificity metric", "Ratio"],
        )
        pattern_metrics, pattern_table = _controlled_evidence_patterns(table)
        alignment_metrics, alignment_table, alignment_chart = _controlled_alignment_outputs(
            discovery_table, table
        )
        choices = [str(feature_id) for feature_id in result.feature_ids]
        candidate_update = gr.update(choices=choices, value=choices[0] if choices else None)
        cross_target_values = _cross_target_shortlist(table, limit=2)
        cross_target_update = gr.update(choices=choices, value=cross_target_values)
        return (
            _candidate_specificity_metrics_markdown(result),
            table,
            chart,
            candidate_update,
            _tsv(table),
            pattern_metrics,
            pattern_table,
            _tsv(pattern_table),
            alignment_metrics,
            alignment_table,
            alignment_chart,
            _tsv(alignment_table),
            cross_target_update,
        )
    except Exception as exc:
        _raise_ui_error(exc)


@gpu(duration=40)
def run_candidate_cross_target_profile(
    prompt: str,
    layer: int,
    token_index: int,
    feature_ids: list[str] | None,
    targets_text: str,
):
    try:
        selected = [int(float(value)) for value in (feature_ids or [])]
        targets = [line for line in str(targets_text).splitlines() if line.strip()]
        result = RUNTIME.candidate_cross_target_profile(
            text=prompt,
            layer=int(layer),
            token_index=int(token_index),
            feature_ids=selected,
            targets=targets,
        )
        columns = [
            "Feature id",
            "Target continuation",
            "Target token count",
            "Native activation",
            "Perturbation L2",
            "Δ mean log p/token",
            "Δ sequence log p",
            "Next-token JS",
        ]
        table = pd.DataFrame(result.rows, columns=columns)
        chart = pd.DataFrame(
            result.chart_rows,
            columns=["Target continuation", "Feature id", "Δ mean log p/token"],
        )
        series_lookup = {feature_id: f"Feature {chr(65 + idx)}" for idx, feature_id in enumerate(result.feature_ids)}
        chart["Series"] = chart["Feature id"].map(lambda value: series_lookup.get(int(value), "Feature"))
        summary_columns = [
            "Feature id",
            "Strongest target",
            "Δ mean log p/token at strongest target",
            "Strongest |effect|",
            "Mean |effect| on other targets",
            "Target-profile ratio",
            "Effect sign pattern",
            "Normalized effect entropy",
            "Effect concentration",
            "Signed bias",
            "Profile pattern",
            "Maximum next-token JS",
        ]
        summary_table = pd.DataFrame(result.summary_rows, columns=summary_columns)
        pairwise_columns = [
            "Feature id",
            "Target A",
            "Target B",
            "Δ normalized preference A−B",
            "|Preference shift|",
            "Direction",
        ]
        pairwise_table = pd.DataFrame(result.pairwise_rows, columns=pairwise_columns)
        pairwise_chart = pairwise_table.copy()
        if not pairwise_chart.empty:
            pairwise_chart["Target pair"] = pairwise_chart["Target A"].astype(str) + " vs " + pairwise_chart["Target B"].astype(str)
            pairwise_chart["Series"] = pairwise_chart["Feature id"].map(
                lambda value: series_lookup.get(int(value), "Feature")
            )
        return (
            _cross_target_metrics_markdown(result),
            table,
            chart,
            summary_table,
            pairwise_table,
            pairwise_chart,
            _tsv(table),
            _tsv(summary_table),
            _tsv(pairwise_table),
        )
    except Exception as exc:
        _raise_ui_error(exc)


@gpu(duration=25)
def run_feature_cue_scan(feature_id: str, layer: int, prompt_stem: str, cue_text: str):
    try:
        if feature_id is None or str(feature_id).strip() == "":
            raise ValueError("Choose a feature id first.")
        cues = [line for line in str(cue_text).splitlines() if line.strip()]
        result = RUNTIME.feature_cue_scan(
            feature_id=int(float(feature_id)),
            layer=int(layer),
            prompt_stem=prompt_stem,
            cues=cues,
        )
        columns = ["Cue", "Full prompt", "Final token", "Activation", "Active in TopK"]
        table = pd.DataFrame(result.rows, columns=columns)
        chart = pd.DataFrame(result.chart_rows, columns=["Cue", "Activation"])
        chart["Series"] = "Cue response"
        return _cue_metrics_markdown(result), table, chart, _tsv(table)
    except Exception as exc:
        _raise_ui_error(exc)


@gpu(duration=30)
def run_feature_cue_context_scan(feature_id: str, layer: int, stems_text: str, cue_text: str):
    try:
        if feature_id is None or str(feature_id).strip() == "":
            raise ValueError("Choose a feature id first.")
        stems = [line for line in str(stems_text).splitlines() if line.strip()]
        cues = [line for line in str(cue_text).splitlines() if line.strip()]
        result = RUNTIME.feature_cue_context_scan(
            feature_id=int(float(feature_id)),
            layer=int(layer),
            stems=stems,
            cues=cues,
        )
        columns = ["Prompt stem", "Cue", "Full prompt", "Final token", "Activation", "Active in TopK"]
        table = pd.DataFrame(result.rows, columns=columns)
        chart = pd.DataFrame(result.chart_rows, columns=["Prompt stem", "Cue", "Activation"])
        return _cue_context_metrics_markdown(result), table, chart, _tsv(table)
    except Exception as exc:
        _raise_ui_error(exc)


def select_candidate_row(table: pd.DataFrame, evt: gr.SelectData):
    if table is None or len(table) == 0:
        return gr.update()
    index = evt.index
    row_index = int(index[0] if isinstance(index, (tuple, list)) else index)
    if row_index < 0 or row_index >= len(table):
        return gr.update()
    value = str(int(float(table.iloc[row_index]["Feature id"])))
    return gr.update(value=value)


def use_candidate_feature(candidate_id: str):
    if candidate_id is None or str(candidate_id).strip() == "":
        raise gr.Error("Run concept-guided discovery and choose a candidate first.")
    value = str(int(float(candidate_id)))
    status = f"Feature {value} loaded for the feature-level experiments."
    return value, value, value, value, status


def mode_help(mode: str):
    if mode == "ablate":
        return gr.update(value=0.0, interactive=False, label="Coefficient (unused for ablation)")
    if mode == "scale":
        return gr.update(value=2.0, interactive=True, label="Feature multiplier")
    return gr.update(value=5.0, interactive=True, label="Additive feature coefficient")


def set_mode_help(mode: str):
    if mode == "ablate":
        return gr.update(value=0.0, interactive=False, label="Multiplier (unused for ablation)")
    return gr.update(value=2.0, interactive=True, label="Shared feature multiplier")


with gr.Blocks(title="FeatureLens — Causal Interpretability Workbench", fill_width=True) as demo:
    gr.HTML(
        '<header class="hero">'
        '<h1>FeatureLens</h1>'
        '<div class="subtitle">Sparse-feature analysis for Qwen3-1.7B and Qwen-Scope SAEs.</div>'
        '</header>'
    )

    global_context = gr.Markdown(
        "**Context:** no prompt location selected yet.",
        elem_classes=["context-card"],
    )

    with gr.Tab("Guide"):
        gr.Markdown(
            "## Working with the app\n"
            "Start in **Workbench** to choose a prompt, layer, and token. Use **Feature evidence** when you "
            "want to find candidates rather than start from a feature id. Return to Workbench or **Feature sets** "
            "for interventions, and use **Paraphrases**, **Layers**, and **Study** for robustness and aggregate evidence.",
            elem_classes=["guide-intro"],
        )
        gr.Markdown(
            "### Evidence, not labels\n"
            "A high activation only says that a feature is present. Association is evaluated separately from causal effect; "
            "random-normalized interventions are the stronger live causal check. Stability metrics show how sensitive a result is to wording or sample choice.\n\n"
            "Dense tables and plots can be focused in place. **Copy TSV** includes the header row."
        )

    with gr.Tab("Workbench"):
        gr.HTML('<div class="section-rule">Inspect</div>')
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=5,
                    value="The derivative of x squared is",
                    placeholder="Enter a prompt to inspect…",
                )
                gr.Examples(
                    examples=[
                        ["The derivative of x squared is"],
                        ["In Python, reverse a list using"],
                        ["Ich möchte einen Tisch für zwei reservieren."],
                        ["I am not fully certain, but the answer may be"],
                    ],
                    inputs=[prompt],
                    label="Examples",
                )
            with gr.Column(scale=3):
                layer = gr.Dropdown(
                    choices=list(SETTINGS.layers),
                    value=SETTINGS.layers[1] if len(SETTINGS.layers) > 1 else SETTINGS.layers[0],
                    label="Residual layer",
                )
                token_index = gr.Number(
                    value=-1,
                    precision=0,
                    label="Prompt token index",
                    info="-1 = final prompt token.",
                )
                top_n = gr.Slider(5, 20, value=12, step=1, label="Displayed active features")
                analyze_btn = gr.Button("Inspect sparse features", variant="primary", elem_classes=["action-btn"])

        gr.Markdown("#### Prompt tokens")
        token_view = gr.HTML(
            '<div class="small-note">Prompt tokens appear here after clicking <b>Inspect sparse features</b>.</div>'
        )
        analysis_metrics = gr.Markdown()
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                _table_heading('Strongest active SAE features')
                feature_table = gr.Dataframe(
                    headers=["Rank", "Feature id", "Activation", "Offline concept hint"],
                    datatype=["number", "number", "number", "str"],
                    interactive=False,
                    label="Strongest active SAE features", show_label=False,
                    wrap=False,
                    max_height=380,
                    buttons=["fullscreen"], elem_classes=["result-table"],
                )
                feature_tsv = gr.Textbox(visible="hidden")
                feature_copy = _copy_button()
            with gr.Column(scale=2):
                feature_plot = gr.BarPlot(
                    x="Feature",
                    y="Activation",
                    color="Series",
                    color_map={"Activation": INK_TEAL},
                    title="Activation profile", elem_id="plot-activation-profile",
                    x_title="Feature id",
                    y_title="Activation",
                    x_label_angle=-35,
                    buttons=["fullscreen", "export"], elem_classes=["fl-plot"],
                    height=330,
                )

        gr.HTML('<div class="section-rule">Single-feature intervention</div>')
        gr.Markdown(
            "Each edit is compared with eight norm-matched random directions. Add a continuation to score the full exact sequence.",
            elem_classes=["section-note"],
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                feature_id = gr.Dropdown(
                    choices=[],
                    allow_custom_value=True,
                    label="Feature id",
                    info="Choose an active feature or enter any valid id.",
                )
                mode = gr.Dropdown(
                    choices=["ablate", "scale", "inject"],
                    value="ablate",
                    label="Single-feature intervention",
                )
                coefficient = gr.Number(
                    value=0.0,
                    interactive=False,
                    label="Coefficient (unused for ablation)",
                )
                target_text = gr.Textbox(
                    label="Target continuation",
                    placeholder="e.g. 2x",
                    info="Leave blank to compare next-token distributions only.",
                )
                max_new = gr.Slider(
                    4,
                    SETTINGS.max_new_tokens,
                    value=min(12, SETTINGS.max_new_tokens),
                    step=1,
                    label="Greedy generation length",
                )
                intervene_btn = gr.Button("Run intervention", variant="primary", elem_classes=["action-btn"])
                intervention_metrics = gr.Markdown()
            with gr.Column(scale=3):
                with gr.Row():
                    baseline_out = gr.Textbox(label="Baseline greedy generation", lines=6, interactive=False)
                    modified_out = gr.Textbox(label="SAE-edited greedy generation", lines=6, interactive=False)
                _table_heading('Next-token distribution shift')
                token_prob_table = gr.Dataframe(
                    interactive=False,
                    label="Next-token distribution shift", show_label=False,
                    buttons=["fullscreen"], elem_classes=["result-table"],
                    wrap=False,
                    max_height=380,
                )
                token_prob_tsv = gr.Textbox(visible="hidden")
                token_prob_copy = _copy_button()
                _table_heading('Target continuation token-by-token score')
                target_token_table = gr.Dataframe(
                    interactive=False,
                    label="Target continuation token-by-token score", show_label=False,
                    buttons=["fullscreen"], elem_classes=["result-table"],
                    wrap=False,
                    max_height=380,
                )
                target_token_tsv = gr.Textbox(visible="hidden")
                target_token_copy = _copy_button()

        gr.HTML('<div class="section-rule">Dose response</div>')
        with gr.Group():
            with gr.Row(equal_height=True):
                dose_feature_id = gr.Dropdown(
                    choices=[],
                    allow_custom_value=True,
                    label="Dose-response feature id",
                    info="Choose an active feature or enter any valid id.",
                    scale=2,
                )
                dose_target_text = gr.Textbox(
                    label="Dose-response target continuation",
                    value="2x",
                    info="Exact continuation scored at every multiplier.",
                    scale=2,
                )
            gr.Markdown(
                "0× ablates the feature; 1× is the no-edit reference; 2× doubles the native coefficient.",
                elem_classes=["section-note"],
            )
            dose_btn = gr.Button("Measure dose response", variant="primary", elem_classes=["action-btn"])
            dose_metrics = gr.Markdown()
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    _table_heading('Scale dose-response measurements')
                    dose_table = gr.Dataframe(
                        interactive=False,
                        label="Scale dose-response measurements", show_label=False,
                        buttons=["fullscreen"], elem_classes=["result-table"],
                        wrap=False,
                    max_height=380,
                    )
                    dose_tsv = gr.Textbox(visible="hidden")
                    dose_copy = _copy_button()
                with gr.Column(scale=2):
                    dose_plot = gr.LinePlot(
                        x="Multiplier",
                        y="Δ mean log p/token",
                        color="Series",
                        color_map={"SAE feature": INK_TEAL},
                        title="Scale dose-response", elem_id="plot-scale-dose-response",
                        x_title="Feature multiplier",
                        y_title="Δ mean log p/token",
                        buttons=["fullscreen", "export"], elem_classes=["fl-plot"],
                        height=330,
                    )

        gr.HTML('<div class="section-rule">Contrastive preference</div>')
        with gr.Group():
            contrastive_feature_id = gr.Dropdown(
                choices=[],
                allow_custom_value=True,
                label="Feature id",
                info="Choose an active feature or enter any valid id.",
            )
            gr.Markdown(
                "Tests whether the edit changes relative preference between two exact continuations, with the same eight-control random baseline.",
                elem_classes=["section-note"],
            )
            with gr.Row(equal_height=True):
                contrastive_a = gr.Textbox(label="Continuation A (preferred)", value="2x", scale=2)
                contrastive_b = gr.Textbox(label="Continuation B (comparison)", value="x", scale=2)
            with gr.Row(equal_height=True):
                contrastive_mode = gr.Dropdown(
                    choices=["ablate", "scale", "inject"],
                    value="ablate",
                    label="Contrastive intervention",
                    scale=1,
                )
                contrastive_coefficient = gr.Number(
                    value=0.0,
                    interactive=False,
                    label="Coefficient (unused for ablation)",
                    scale=1,
                )
            contrastive_btn = gr.Button(
                "Compare continuation preference", variant="primary", elem_classes=["action-btn"]
            )
            contrastive_metrics = gr.Markdown()
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    _table_heading('Contrastive continuation scores')
                    contrastive_table = gr.Dataframe(
                        interactive=False,
                        label="Contrastive continuation scores", show_label=False,
                        buttons=["fullscreen"], elem_classes=["result-table"],
                        wrap=False,
                        max_height=320,
                    )
                    contrastive_tsv = gr.Textbox(visible="hidden")
                    contrastive_copy = _copy_button()
                with gr.Column(scale=2):
                    contrastive_plot = gr.BarPlot(
                        x="Condition",
                        y="A−B sequence log-odds",
                        color="Series",
                        color_map={"Contrastive preference": INK_TEAL},
                        title="Preference between exact continuations", elem_id="plot-contrastive-preference",
                        x_title="Execution condition",
                        y_title="Sequence log-odds A−B",
                        buttons=["fullscreen", "export"], elem_classes=["fl-plot"],
                        height=320,
                    )

    with gr.Tab("Feature sets"):
        gr.Markdown(
            "## Feature sets\n"
            "Jointly edit active SAE features at the current Workbench location.",
            elem_classes=["guide-intro"],
        )
        feature_set_location = gr.Markdown("", visible=False)
        feature_set_ids = gr.Dropdown(
            choices=[],
            value=[],
            multiselect=True,
            allow_custom_value=True,
            max_choices=12,
            label="Feature set",
        )

        gr.HTML('<div class="section-rule">Joint intervention</div>')
        gr.HTML(
            '<div class="instrument-note">Ablation removes each selected feature at its native coefficient. Scale applies one shared multiplier before the decoder deltas are summed.</div>'
        )
        with gr.Row(equal_height=True):
            set_mode = gr.Dropdown(
                choices=["ablate", "scale"],
                value="ablate",
                label="Intervention",
                scale=1,
            )
            set_coefficient = gr.Number(
                value=0.0,
                interactive=False,
                label="Multiplier (unused for ablation)",
                scale=1,
            )
            set_target = gr.Textbox(
                label="Target continuation",
                value="2x",
                lines=1,
                scale=2,
            )
        set_btn = gr.Button("Run joint intervention", variant="primary", elem_classes=["action-btn"])
        set_metrics = gr.Markdown()
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                _table_heading('Joint intervention features')
                set_feature_table = gr.Dataframe(
                    interactive=False,
                    label="Joint intervention features", show_label=False,
                    buttons=["fullscreen"], elem_classes=["result-table"],
                    wrap=False,
                    max_height=380,
                )
                set_feature_tsv = gr.Textbox(visible="hidden")
                set_feature_copy = _copy_button()
            with gr.Column(scale=3):
                _table_heading('Target continuation token-by-token score')
                set_target_table = gr.Dataframe(
                    interactive=False,
                    label="Target continuation token-by-token score", show_label=False,
                    buttons=["fullscreen"], elem_classes=["result-table"],
                    wrap=False,
                    max_height=380,
                )
                set_target_tsv = gr.Textbox(visible="hidden")
                set_target_copy = _copy_button()

        gr.HTML('<div class="section-rule">Set-size sensitivity</div>')
        gr.Markdown(
            "Ablate the 1, 3, and 5 strongest active features with matched random controls.",
            elem_classes=["section-note"],
        )
        set_sweep_target = gr.Textbox(label="Target continuation for set-size sweep", value="2x")
        set_sweep_btn = gr.Button("Run 1/3/5-feature ablation sweep", variant="primary", elem_classes=["action-btn"])
        set_sweep_note = gr.Markdown()
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                _table_heading('Feature-set size measurements')
                set_sweep_table = gr.Dataframe(
                    interactive=False,
                    label="Feature-set size measurements", show_label=False,
                    buttons=["fullscreen"], elem_classes=["result-table"],
                    wrap=False,
                    max_height=380,
                )
                set_sweep_tsv = gr.Textbox(visible="hidden")
                set_sweep_copy = _copy_button()
            with gr.Column(scale=2):
                set_sweep_plot = gr.LinePlot(
                    x="Set size k",
                    y="Δ mean log p/token",
                    color="Condition",
                    color_map={
                        "Top-k SAE ablation": INK_TEAL,
                        "Random signed mean": INK_STONE,
                    },
                    title="Effect vs feature-set size", elem_id="plot-feature-set-size",
                    x_title="Number of jointly ablated features",
                    y_title="Δ mean log p/token",
                    buttons=["fullscreen", "export"], elem_classes=["fl-plot"],
                    height=330,
                )

        gr.HTML('<div class="section-rule">Interaction decomposition</div>')
        gr.Markdown(
            "Compare the joint ablation with the sum of individual effects.",
            elem_classes=["section-note"],
        )
        interaction_target = gr.Textbox(label="Target continuation for interaction test", value="2x")
        interaction_btn = gr.Button("Run individual-vs-joint decomposition", variant="primary", elem_classes=["action-btn"])
        interaction_metrics = gr.Markdown()
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                _table_heading('Individual and joint ablation measurements')
                interaction_table = gr.Dataframe(
                    interactive=False,
                    label="Individual and joint ablation measurements", show_label=False,
                    buttons=["fullscreen"], elem_classes=["result-table"],
                    wrap=False,
                    max_height=380,
                )
                interaction_tsv = gr.Textbox(visible="hidden")
                interaction_copy = _copy_button()
            with gr.Column(scale=2):
                interaction_plot = gr.BarPlot(
                    x="Condition",
                    y="Δ mean log p/token",
                    color="Series",
                    color_map={"Ablation effect": INK_UMBER},
                    title="Individual vs joint effect", elem_id="plot-individual-vs-joint",
                    x_title="Intervention condition",
                    y_title="Δ mean log p/token",
                    x_label_angle=-25,
                    buttons=["fullscreen", "export"], elem_classes=["fl-plot"],
                    height=330,
                )


        gr.HTML('<div class="section-rule">Decoder geometry</div>')
        with gr.Group():
            gr.Markdown(
                "Decoder-vector cosines and joint-edit geometry.",
                elem_classes=["section-note"],
            )
            geometry_btn = gr.Button(
                "Inspect decoder geometry", variant="primary", elem_classes=["action-btn"]
            )
            geometry_metrics = gr.Markdown()
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    _table_heading('Pairwise decoder geometry')
                    geometry_table = gr.Dataframe(
                        interactive=False,
                        label="Pairwise decoder geometry", show_label=False,
                        buttons=["fullscreen"], elem_classes=["result-table"],
                        wrap=False,
                        max_height=340,
                    )
                    geometry_tsv = gr.Textbox(visible="hidden")
                    geometry_copy = _copy_button()
                with gr.Column(scale=2):
                    geometry_plot = gr.BarPlot(
                        x="Feature pair",
                        y="Decoder cosine",
                        color="Series",
                        color_map={"Decoder cosine": INK_PLUM},
                        title="Pairwise SAE decoder cosine", elem_id="plot-decoder-geometry",
                        x_title="Feature pair",
                        y_title="Cosine similarity",
                        x_label_angle=-30,
                        buttons=["fullscreen", "export"], elem_classes=["fl-plot"],
                        height=320,
                    )

    with gr.Tab("Features"):
        gr.Markdown(
            "## Feature evidence\n"
            "Discover concept-associated candidates, screen causal effects, then inspect individual features in context.",
            elem_classes=["guide-intro"],
        )
        gr.HTML('<div class="section-rule">Candidate discovery</div>')
        gr.Markdown(
            "Rank SAE features from the controlled prompt set. **Causal-ready at current token** keeps only candidates active at the selected Workbench location.",
            elem_classes=["section-note"],
        )
        with gr.Row(equal_height=True):
            discovery_concept = gr.Dropdown(
                choices=[
                    "code", "mathematics", "positive_sentiment", "negative_sentiment",
                    "german_language", "factual_entities", "uncertainty"
                ],
                value="mathematics",
                label="Target concept",
            )
            discovery_layer = gr.Dropdown(choices=list(SETTINGS.layers), value=SETTINGS.layers[1], label="Residual layer")
            discovery_n = gr.Slider(2, 6, value=SETTINGS.contrast_prompts_per_concept, step=1, label="Prompts per concept")
            discovery_top_n = gr.Slider(5, 20, value=12, step=1, label="Candidate features")
        with gr.Row(equal_height=True):
            discovery_ranking = gr.Dropdown(
                choices=["Balanced selectivity", "Causal-ready at current token", "Raw mean difference"],
                value="Balanced selectivity",
                label="Candidate ranking",
                info="Balanced selectivity finds concept-associated candidates; Causal-ready restricts to features active at the selected Workbench token; raw mean difference exposes scale-dominated ranking.",
                scale=2,
            )
            gr.Markdown(
                "Controlled prompts and the selected Workbench location share one batch.",
                elem_classes=["section-note"],
            )
        discovery_btn = gr.Button("Rank candidates", variant="primary", elem_classes=["action-btn"])
        discovery_metrics = gr.Markdown()
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                _table_heading('Candidate feature evidence')
                discovery_table = gr.Dataframe(
                    interactive=False, label="Candidate feature evidence", show_label=False, buttons=["fullscreen"], elem_classes=["result-table"],
                    wrap=False, max_height=420
                )
                discovery_tsv = gr.Textbox(visible="hidden")
                discovery_copy = _copy_button()
            with gr.Column(scale=2):
                discovery_plot = gr.BarPlot(
                    x="Feature", y="Candidate score", color="Series",
                    color_map={"Candidate score": INK_TEAL}, title="Candidate evidence score", elem_id="plot-candidate-discovery",
                    x_title="Feature id", y_title="Exploratory ranking score", x_label_angle=-35,
                    buttons=["fullscreen", "export"], elem_classes=["fl-plot"], height=330
                )
        gr.Markdown(
            "Click a row to select that feature.",
            elem_classes=["candidate-help"],
        )
        with gr.Row(equal_height=True):
            discovery_candidate = gr.Dropdown(choices=[], label="Selected candidate feature id", allow_custom_value=True, scale=3)
            use_candidate_btn = gr.Button(
                "Use selected feature", variant="secondary", elem_classes=["copy-btn"], scale=2
            )
        candidate_use_status = gr.Markdown()

        gr.HTML('<div class="section-rule">Candidate triage</div>')
        gr.Markdown(
            "Ablate several candidates in one batch before spending random controls on a smaller shortlist.",
            elem_classes=["section-note"],
        )
        with gr.Row(equal_height=True):
            candidate_screen_ids = gr.Dropdown(
                choices=[],
                value=[],
                multiselect=True,
                allow_custom_value=True,
                max_choices=8,
                label="Candidate features to screen",
                info="Populated by concept-guided discovery; up to eight features per batch.",
                scale=3,
            )
            candidate_screen_target = gr.Textbox(
                label="Screen target continuation",
                value="2x",
                info="Exact continuation used only for this screening run.",
                scale=2,
            )
        candidate_screen_btn = gr.Button(
            "Screen ablations", variant="primary", elem_classes=["action-btn"]
        )
        candidate_screen_metrics = gr.Markdown()
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                _table_heading('Candidate ablation screen')
                candidate_screen_table = gr.Dataframe(
                    interactive=False,
                    label="Candidate ablation screen",
                    show_label=False,
                    buttons=["fullscreen"],
                    elem_classes=["result-table"],
                    wrap=False,
                    max_height=380,
                )
                candidate_screen_tsv = gr.Textbox(visible="hidden")
                candidate_screen_copy = _copy_button()
            with gr.Column(scale=2):
                candidate_screen_plot = gr.BarPlot(
                    x="Feature",
                    y="Δ mean log p/token",
                    color="Series",
                    color_map={"Candidate ablation": INK_TEAL},
                    title="Candidate ablation target effect",
                    elem_id="plot-candidate-causal-screen",
                    x_title="Feature id",
                    y_title="Δ mean log p/token",
                    x_label_angle=-35,
                    buttons=["fullscreen", "export"],
                    elem_classes=["fl-plot"],
                    height=330,
                )
        gr.Markdown(
            "Click a row to select that feature.",
            elem_classes=["candidate-help"],
        )

        gr.Markdown("#### Association vs causal influence")
        gr.Markdown(
            "Discovery rank versus target-effect and distribution-shift rank for the same shortlist.",
            elem_classes=["candidate-help"],
        )
        candidate_alignment_metrics = gr.Markdown()
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                _table_heading('Discovery–causality alignment')
                candidate_alignment_table = gr.Dataframe(
                    interactive=False,
                    label="Discovery–causality alignment",
                    show_label=False,
                    buttons=["fullscreen"],
                    elem_classes=["result-table"],
                    wrap=False,
                    max_height=360,
                )
                candidate_alignment_tsv = gr.Textbox(visible="hidden")
                candidate_alignment_copy = _copy_button()
            with gr.Column(scale=2):
                candidate_alignment_plot = gr.ScatterPlot(
                    x="Candidate score",
                    y="|Δ mean log p/token|",
                    color="Series",
                    color_map={"Screened candidate": INK_TEAL},
                    title="Association evidence vs target effect",
                    elem_id="plot-association-causality",
                    x_title="Discovery candidate score",
                    y_title="|Δ mean log p/token|",
                    tooltip=[
                        "Feature id",
                        "Discovery rank",
                        "Target-effect rank",
                        "Next-token JS",
                    ],
                    buttons=["fullscreen", "export"],
                    elem_classes=["fl-plot"],
                    height=330,
                )

        gr.Markdown("#### Controlled candidate specificity")
        gr.Markdown(
            "Up to three candidates, each with its own eight-direction norm-matched random ensemble.",
            elem_classes=["section-note"],
        )
        with gr.Row(equal_height=True):
            candidate_specificity_ids = gr.Dropdown(
                choices=[],
                value=[],
                multiselect=True,
                allow_custom_value=True,
                max_choices=3,
                label="Candidates for controlled comparison",
                info="Auto-filled after triage; choose up to three features.",
                scale=3,
            )
            candidate_specificity_target = gr.Textbox(
                label="Controlled target continuation",
                value="2x",
                info="Exact continuation scored for every candidate and its random controls.",
                scale=2,
            )
        candidate_specificity_btn = gr.Button(
            "Run controlled comparison", variant="primary", elem_classes=["action-btn"]
        )
        candidate_specificity_metrics = gr.Markdown()
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                _table_heading('Controlled candidate specificity')
                candidate_specificity_table = gr.Dataframe(
                    interactive=False,
                    label="Controlled candidate specificity",
                    show_label=False,
                    buttons=["fullscreen"],
                    elem_classes=["result-table"],
                    wrap=False,
                    max_height=390,
                )
                candidate_specificity_tsv = gr.Textbox(visible="hidden")
                candidate_specificity_copy = _copy_button()
            with gr.Column(scale=2):
                candidate_specificity_plot = gr.BarPlot(
                    x="Feature",
                    y="Ratio",
                    color="Specificity metric",
                    color_map={
                        "Target specificity": INK_TEAL,
                        "JS specificity": INK_UMBER,
                    },
                    title="Random-normalized causal specificity",
                    elem_id="plot-candidate-specificity",
                    x_title="Feature id",
                    y_title="SAE effect / random mean effect",
                    x_label_angle=-35,
                    buttons=["fullscreen", "export"],
                    elem_classes=["fl-plot"],
                    height=330,
                )

        gr.Markdown("#### Controlled evidence patterns")
        gr.Markdown(
            "Compact reading of target versus distributional specificity.",
            elem_classes=["candidate-help"],
        )
        controlled_pattern_metrics = gr.Markdown()
        _table_heading('Controlled evidence pattern summary')
        controlled_pattern_table = gr.Dataframe(
            interactive=False,
            label="Controlled evidence pattern summary",
            show_label=False,
            buttons=["fullscreen"],
            elem_classes=["result-table"],
            wrap=True,
            max_height=300,
        )
        controlled_pattern_tsv = gr.Textbox(visible="hidden")
        controlled_pattern_copy = _copy_button()

        gr.Markdown("#### Association vs controlled causality")
        gr.Markdown(
            "Concept evidence versus random-normalized causal specificity for the same candidates.",
            elem_classes=["candidate-help"],
        )
        controlled_alignment_metrics = gr.Markdown()
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                _table_heading('Discovery–controlled-causality alignment')
                controlled_alignment_table = gr.Dataframe(
                    interactive=False,
                    label="Discovery–controlled-causality alignment",
                    show_label=False,
                    buttons=["fullscreen"],
                    elem_classes=["result-table"],
                    wrap=False,
                    max_height=360,
                )
                controlled_alignment_tsv = gr.Textbox(visible="hidden")
                controlled_alignment_copy = _copy_button()
            with gr.Column(scale=2):
                controlled_alignment_plot = gr.ScatterPlot(
                    x="Candidate score",
                    y="Target specificity ratio",
                    color="Series",
                    color_map={"Controlled candidate": INK_TEAL},
                    title="Association evidence vs controlled target specificity",
                    elem_id="plot-controlled-association-causality",
                    x_title="Discovery candidate score",
                    y_title="Target specificity ratio",
                    tooltip=[
                        "Feature id",
                        "Discovery rank",
                        "Specificity rank",
                        "Target empirical tail p",
                    ],
                    buttons=["fullscreen", "export"],
                    elem_classes=["fl-plot"],
                    height=330,
                )

        gr.HTML('<div class="section-rule">Cross-target profile</div>')
        gr.Markdown(
            "Profile the same native ablation across several exact continuations; random controls are not used in this screen.",
            elem_classes=["section-note"],
        )
        with gr.Row(equal_height=True):
            cross_target_ids = gr.Dropdown(
                choices=[],
                value=[],
                multiselect=True,
                allow_custom_value=True,
                max_choices=3,
                label="Features for cross-target profile",
                info="Auto-filled from the target-specificity leader and JS-specificity leader when available.",
                scale=3,
            )
            cross_target_text = gr.Textbox(
                label="Exact target continuations (one per line)",
                value="2x\nx\n0\nx^2",
                lines=4,
                info="Two to five continuations. Each is teacher-forced separately.",
                scale=2,
            )
        cross_target_btn = gr.Button(
            "Profile target effects", variant="primary", elem_classes=["action-btn"]
        )
        cross_target_metrics = gr.Markdown()
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                _table_heading('Cross-target causal profile')
                cross_target_table = gr.Dataframe(
                    interactive=False,
                    label="Cross-target causal profile",
                    show_label=False,
                    buttons=["fullscreen"],
                    elem_classes=["result-table"],
                    wrap=False,
                    max_height=390,
                )
                cross_target_tsv = gr.Textbox(visible="hidden")
                cross_target_copy = _copy_button()
            with gr.Column(scale=2):
                cross_target_plot = gr.BarPlot(
                    x="Target continuation",
                    y="Δ mean log p/token",
                    color="Series",
                    color_map={"Feature A": INK_TEAL, "Feature B": INK_UMBER, "Feature C": INK_PLUM},
                    title="Candidate effect across exact continuations",
                    elem_id="plot-cross-target-profile",
                    x_title="Target continuation",
                    y_title="Δ mean log p/token",
                    buttons=["fullscreen", "export"],
                    elem_classes=["fl-plot"],
                    height=330,
                )
        _table_heading('Target-profile summary')
        cross_target_summary_table = gr.Dataframe(
            interactive=False,
            label="Target-profile summary",
            show_label=False,
            buttons=["fullscreen"],
            elem_classes=["result-table"],
            wrap=False,
            max_height=260,
        )
        cross_target_summary_tsv = gr.Textbox(visible="hidden")
        cross_target_summary_copy = _copy_button()

        gr.Markdown("#### Pairwise target preference shifts")
        gr.Markdown(
            "From the same target scores. Positive Δ(A−B) shifts normalized preference toward A.",
            elem_classes=["section-note"],
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                _table_heading('Pairwise target preference shifts')
                cross_target_pairwise_table = gr.Dataframe(
                    interactive=False,
                    label="Pairwise target preference shifts",
                    show_label=False,
                    buttons=["fullscreen"],
                    elem_classes=["result-table"],
                    wrap=False,
                    max_height=340,
                )
                cross_target_pairwise_tsv = gr.Textbox(visible="hidden")
                cross_target_pairwise_copy = _copy_button()
            with gr.Column(scale=2):
                cross_target_pairwise_plot = gr.BarPlot(
                    x="Target pair",
                    y="Δ normalized preference A−B",
                    color="Series",
                    color_map={"Feature A": INK_TEAL, "Feature B": INK_UMBER, "Feature C": INK_PLUM},
                    title="Pairwise target preference shifts",
                    elem_id="plot-cross-target-pairwise",
                    x_title="Target pair",
                    y_title="Δ normalized preference A−B",
                    x_label_angle=-30,
                    buttons=["fullscreen", "export"],
                    elem_classes=["fl-plot"],
                    height=330,
                )

        gr.HTML('<div class="section-rule">Inspect one feature</div>')
        contrast_location = gr.Markdown("", visible=False)
        with gr.Row(equal_height=True):
            contrast_feature_id = gr.Dropdown(
                choices=[],
                allow_custom_value=True,
                label="Feature id",
                scale=2,
            )
            contrast_layer = gr.Dropdown(
                choices=list(SETTINGS.layers),
                value=SETTINGS.layers[1],
                label="Residual layer",
                scale=1,
            )
            contrast_n = gr.Slider(
                2,
                6,
                value=SETTINGS.contrast_prompts_per_concept,
                step=1,
                label="Prompts per concept",
                scale=2,
            )

        gr.Markdown("### Activation trace")
        gr.Markdown(
            "Activation of the selected feature across prompt tokens.",
            elem_classes=["section-note"],
        )
        trace_btn = gr.Button("Trace feature", variant="primary", elem_classes=["action-btn"])
        trace_metrics = gr.Markdown()
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                _table_heading('Feature activation by prompt token')
                trace_table = gr.Dataframe(
                    interactive=False,
                    label="Feature activation by prompt token", show_label=False,
                    buttons=["fullscreen"], elem_classes=["result-table"],
                    wrap=False,
                    max_height=340,
                )
                trace_tsv = gr.Textbox(visible="hidden")
                trace_copy = _copy_button()
            with gr.Column(scale=2):
                trace_plot = gr.BarPlot(
                    x="Token",
                    y="Activation",
                    color="Series",
                    color_map={"Feature activation": INK_TEAL},
                    title="Feature activation across prompt tokens", elem_id="plot-feature-token-trace",
                    x_title="Prompt token",
                    y_title="Activation",
                    x_label_angle=-35,
                    buttons=["fullscreen", "export"], elem_classes=["fl-plot"],
                    height=320,
                )

        gr.HTML('<div class="section-rule">Completion-cue sensitivity</div>')
        gr.Markdown(
            "Final-token response to alternative completion cues.",
            elem_classes=["section-note"],
        )
        with gr.Row(equal_height=True):
            cue_stem = gr.Textbox(label="Prompt stem", value="The derivative of x squared", lines=2, scale=3)
            cue_text = gr.Textbox(label="Completion cues (one per line)", value="is\n=\n:\nequals\ntherefore", lines=5, scale=2)
        cue_btn = gr.Button("Compare completion cues", variant="primary", elem_classes=["action-btn"])
        cue_metrics = gr.Markdown()
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                _table_heading('Feature response by completion cue')
                cue_table = gr.Dataframe(interactive=False, label="Feature response by completion cue", show_label=False, buttons=["fullscreen"], elem_classes=["result-table"], wrap=False, max_height=340)
                cue_tsv = gr.Textbox(visible="hidden")
                cue_copy = _copy_button()
            with gr.Column(scale=2):
                cue_plot = gr.BarPlot(
                    x="Cue", y="Activation", color="Series", color_map={"Cue response": INK_UMBER},
                    title="Completion-cue feature response", elem_id="plot-completion-cue-response", x_title="Cue", y_title="Final-token activation",
                    buttons=["fullscreen", "export"], elem_classes=["fl-plot"], height=320
                )

        gr.HTML('<div class="section-rule">Cue × context specificity</div>')
        gr.Markdown(
            "Cross the same cues with unrelated prompt stems.",
            elem_classes=["section-note"],
        )
        with gr.Row(equal_height=True):
            cue_context_stems = gr.Textbox(
                label="Prompt stems (one per line)",
                value="The derivative of x squared\nThe capital of Germany\nThe weather today\nMy name",
                lines=5,
                scale=3,
            )
            cue_context_cues = gr.Textbox(
                label="Completion cues (one per line)",
                value="is\n=\n:\nequals\ntherefore",
                lines=5,
                scale=2,
            )
        cue_context_btn = gr.Button("Run cue × context", variant="primary", elem_classes=["action-btn"])
        cue_context_metrics = gr.Markdown()
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                _table_heading('Cue × context feature response')
                cue_context_table = gr.Dataframe(
                    interactive=False,
                    label="Cue × context feature response", show_label=False,
                    buttons=["fullscreen"],
                    elem_classes=["result-table"],
                    wrap=False,
                    max_height=420,
                )
                cue_context_tsv = gr.Textbox(visible="hidden")
                cue_context_copy = _copy_button()
            with gr.Column(scale=2):
                cue_context_plot = gr.BarPlot(
                    x="Prompt stem",
                    y="Activation",
                    color="Cue",
                    color_map={
                        "is": INK_TEAL,
                        "=": INK_UMBER,
                        ":": INK_RED,
                        "equals": INK_PLUM,
                        "therefore": INK_STONE,
                    },
                    title="Cue response across contexts",
                    elem_id="plot-cue-context-matrix",
                    x_title="Prompt stem",
                    y_title="Final-token activation",
                    x_label_angle=-25,
                    buttons=["fullscreen", "export"],
                    elem_classes=["fl-plot"],
                    height=340,
                )

        gr.HTML('<div class="section-rule">Controlled concept contrast</div>')
        gr.Markdown("Prompt-wide max activation across the controlled concept set.", elem_classes=["section-note"])
        contrast_btn = gr.Button("Compare concepts", variant="primary", elem_classes=["action-btn"])
        contrast_metrics = gr.Markdown()
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                _table_heading('Feature activation by controlled concept')
                contrast_table = gr.Dataframe(
                    interactive=False,
                    label="Feature activation by controlled concept", show_label=False,
                    buttons=["fullscreen"], elem_classes=["result-table"],
                    wrap=False,
                    max_height=380,
                )
                contrast_tsv = gr.Textbox(visible="hidden")
                contrast_copy = _copy_button()
            with gr.Column(scale=2):
                contrast_plot = gr.BarPlot(
                    x="Concept",
                    y="Mean prompt-wide max",
                    color="Series",
                    color_map={"Prompt-wide max": INK_BLUEGREY},
                    title="Prompt-wide controlled concept contrast", elem_id="plot-controlled-concept-contrast",
                    x_title="Concept",
                    y_title="Mean max activation",
                    x_label_angle=-25,
                    buttons=["fullscreen", "export"], elem_classes=["fl-plot"],
                    height=330,
                )

    with gr.Tab("Paraphrases"):
        gr.Markdown(
            "## Paraphrase robustness\n"
            "Compare selected-token features with prompt-wide max-pooled SAE profiles.",
            elem_classes=["guide-intro"],
        )
        with gr.Row():
            para_a = gr.Textbox(
                label="Original prompt",
                lines=4,
                value="The derivative of x squared is",
            )
            para_b = gr.Textbox(
                label="Paraphrase",
                lines=4,
                value="Differentiate x squared with respect to x:",
            )
        with gr.Row():
            para_layer = gr.Dropdown(
                choices=list(SETTINGS.layers),
                value=SETTINGS.layers[1],
                label="Residual layer",
            )
            para_idx_a = gr.Number(value=-1, precision=0, label="Original prompt token index")
            para_idx_b = gr.Number(value=-1, precision=0, label="Paraphrase token index")
            para_top_n = gr.Slider(5, 20, value=12, step=1, label="Displayed active features")
        para_btn = gr.Button("Compare representations", variant="primary", elem_classes=["action-btn"])
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Original prompt tokens")
                para_tokens_a = gr.HTML()
            with gr.Column():
                gr.Markdown("#### Paraphrase tokens")
                para_tokens_b = gr.HTML()
        para_metrics = gr.Markdown()
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                _table_heading('Top-feature overlap at selected tokens')
                para_table = gr.Dataframe(
                    interactive=False,
                    label="Top-feature overlap at selected tokens", show_label=False,
                    buttons=["fullscreen"], elem_classes=["result-table"],
                    wrap=False,
                    max_height=380,
                )
                para_tsv = gr.Textbox(visible="hidden")
                para_copy = _copy_button()
            with gr.Column(scale=2):
                para_plot = gr.BarPlot(
                    x="Feature",
                    y="Activation",
                    color="Prompt",
                    color_map={"Original": INK_TEAL, "Paraphrase": INK_PLUM},
                    title="Selected-token feature activations", elem_id="plot-paraphrase-selected-token",
                    x_title="Feature id",
                    y_title="Activation",
                    x_label_angle=-35,
                    buttons=["fullscreen", "export"], elem_classes=["fl-plot"],
                    height=330,
                )

    with gr.Tab("Layers"):
        gr.Markdown(
            "## Layer trajectory\n"
            "Compare reconstruction and sparse-activation structure across the selected early, middle, and late SAE layers.",
            elem_classes=["guide-intro"],
        )
        with gr.Row():
            trajectory_prompt = gr.Textbox(
                label="Prompt",
                lines=5,
                value="The derivative of x squared is",
                scale=4,
            )
            trajectory_token = gr.Number(
                value=-1,
                precision=0,
                label="Prompt token index",
                info="-1 = final token",
                scale=1,
            )
        trajectory_btn = gr.Button("Compare layers", variant="primary", elem_classes=["action-btn"])
        gr.Markdown("#### Prompt tokens")
        trajectory_tokens = gr.HTML()
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                _table_heading('Layer diagnostics')
                trajectory_table = gr.Dataframe(
                    interactive=False,
                    label="Layer diagnostics", show_label=False,
                    buttons=["fullscreen"], elem_classes=["result-table"],
                    wrap=False,
                    max_height=380,
                )
                trajectory_tsv = gr.Textbox(visible="hidden")
                trajectory_copy = _copy_button()
            with gr.Column(scale=2):
                trajectory_plot = gr.LinePlot(
                    x="Layer",
                    y="Value",
                    color="Metric",
                    color_map={
                        "Reconstruction cosine": INK_TEAL,
                        "Top-5 mass": INK_UMBER,
                        "Activation entropy": INK_RED,
                    },
                    title="Representation trajectory", elem_id="plot-layer-trajectory",
                    x_title="Layer",
                    y_title="Normalized value",
                    buttons=["fullscreen", "export"], elem_classes=["fl-plot"],
                    height=330,
                )

    with gr.Tab("Study"):
        gr.Markdown(STUDY.overview_markdown())
        gr.Markdown(STUDY.readiness_markdown(), elem_classes=["small-note"])

        offline_study = STUDY.dataframe("study_feature_summary.csv")
        offline_stability = STUDY.dataframe("selection_stability.csv")
        offline_positions = STUDY.dataframe("causal_position_summary.csv")
        offline_layers = STUDY.dataframe("layer_metrics.csv")

        if not offline_study.empty:
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    _table_heading("Selected feature evidence by concept")
                    gr.Dataframe(
                        value=offline_study,
                        interactive=False,
                        show_label=False,
                        buttons=["fullscreen"],
                        elem_classes=["result-table"],
                        wrap=False,
                        max_height=420,
                    )
                with gr.Column(scale=2):
                    gr.Image(
                        value=STUDY.figure("association_vs_causality.png"),
                        label="Association evidence vs causal specificity",
                        interactive=False,
                        show_label=True,
                        height=360,
                    )

            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    _table_heading("Causal position sensitivity")
                    position_preview = offline_positions[offline_positions["concept"] == "__all__"] if not offline_positions.empty else offline_positions
                    gr.Dataframe(
                        value=position_preview,
                        interactive=False,
                        show_label=False,
                        buttons=["fullscreen"],
                        elem_classes=["result-table"],
                        wrap=False,
                        max_height=320,
                    )
                with gr.Column(scale=2):
                    gr.Image(
                        value=STUDY.figure("causal_position_sensitivity.png"),
                        label="Final-token vs max-active intervention",
                        interactive=False,
                        show_label=True,
                        height=360,
                    )

            _table_heading("Candidate selection stability")
            stability_preview = offline_stability.sort_values(
                ["resample_support", "full_score"], ascending=[False, False]
            ).head(40)
            gr.Dataframe(
                value=stability_preview,
                interactive=False,
                show_label=False,
                buttons=["fullscreen"],
                elem_classes=["result-table"],
                wrap=False,
                max_height=360,
            )

            _table_heading("Layer diagnostics")
            gr.Dataframe(
                value=offline_layers,
                interactive=False,
                show_label=False,
                buttons=["fullscreen"],
                elem_classes=["result-table"],
                wrap=False,
                max_height=300,
            )

            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.Image(
                        value=STUDY.figure("feature_auroc.png"),
                        label="Held-out sparse-feature AUROC",
                        interactive=False,
                        show_label=True,
                        height=360,
                    )
                with gr.Column(scale=1):
                    gr.Image(
                        value=STUDY.figure("feature_set_effects.png"),
                        label="Feature-set causal effects",
                        interactive=False,
                        show_label=True,
                        height=360,
                    )
        else:
            gr.Markdown(
                "Run the offline study to populate measured tables and figures. Until then, this tab stays intentionally empty."
            )

    with gr.Tab("Method"):
        gr.Markdown(
        r"""
### Reconstruction-preserving intervention

For residual vector $h$, sparse coefficient $z_i$, decoder direction $d_i$, and scale $\alpha$:

- **Ablate:** $h' = h - z_i d_i$
- **Scale:** $h' = h + (\alpha - 1) z_i d_i$
- **Inject:** $h' = h + \delta d_i$

For a feature set $S$:

$$
h' = h + \sum_{i \in S} \Delta z_i d_i
$$

FeatureLens patches the intervention delta into the **original residual**; it does not replace the residual with the full SAE reconstruction.

### Control discipline

Batched experiments include an explicit **zero-edit reference**. Causal effects are measured against that condition rather than against a separately executed baseline, avoiding batch-versus-single numerical drift.

Random specificity uses norm-matched residual directions so that SAE interventions are compared against perturbations with the same $L_2$ magnitude.

### Causal position

The offline study evaluates two intervention policies:

- **Final token:** intervene at the final prompt-token residual.
- **Max-active token:** intervene at the prompt position where the selected SAE feature has maximum activation,

$$
t^* = \arg\max_t z_f(t)
$$

where $z_f(t)$ is the activation of selected feature $f$ at token position $t$.

The intervention location is chosen only from SAE activation; behavioral outcomes are not used to select the token.

### Statistical unit

For the offline causal study, the **causal task** is the primary statistical unit.

Ablation and amplification effects are first aggregated within each task before paired bootstrap and sign-flip inference. This avoids treating two interventions on the same prompt as independent observations.

### Evidence ladder

1. SAE reconstruction quality.
2. Held-out feature/concept prediction.
3. Candidate-selection stability.
4. Local and prompt-wide paraphrase robustness.
5. Single-feature causal intervention and dose response.
6. Contrastive continuation preference.
7. Joint feature-set intervention and interaction analysis.
8. Specificity relative to norm-matched random controls.
9. Final-token versus max-active causal-position sensitivity.

Association, robustness, geometry, and causal intervention are treated as distinct forms of evidence.
""",
        latex_delimiters=LATEX_DELIMITERS,
    )

    gr.HTML('<div class="bottom-spacer" aria-hidden="true"></div>')

    demo.load(fn=None, js=INSTALL_REFLOW_JS, queue=False)

    # Event wiring.
    analyze_btn.click(
        analyze_prompt,
        inputs=[prompt, layer, token_index, top_n],
        outputs=[
            token_view,
            feature_table,
            feature_plot,
            feature_id,
            dose_feature_id,
            contrastive_feature_id,
            feature_set_ids,
            contrast_feature_id,
            contrast_layer,
            analysis_metrics,
            feature_set_location,
            contrast_location,
            global_context,
            feature_tsv,
        ],
    )
    mode.change(mode_help, inputs=[mode], outputs=[coefficient])
    intervene_btn.click(
        run_intervention,
        inputs=[prompt, layer, token_index, feature_id, mode, coefficient, target_text, max_new],
        outputs=[
            baseline_out,
            modified_out,
            intervention_metrics,
            token_prob_table,
            target_token_table,
            token_prob_tsv,
            target_token_tsv,
        ],
    )
    dose_btn.click(
        run_dose_response,
        inputs=[prompt, layer, token_index, dose_feature_id, dose_target_text],
        outputs=[dose_table, dose_plot, dose_metrics, dose_tsv],
    )
    contrastive_mode.change(mode_help, inputs=[contrastive_mode], outputs=[contrastive_coefficient])
    contrastive_btn.click(
        run_contrastive_causal,
        inputs=[prompt, layer, token_index, contrastive_feature_id, contrastive_mode, contrastive_coefficient, contrastive_a, contrastive_b],
        outputs=[contrastive_metrics, contrastive_table, contrastive_plot, contrastive_tsv],
    )
    set_mode.change(set_mode_help, inputs=[set_mode], outputs=[set_coefficient])
    set_btn.click(
        run_feature_set,
        inputs=[prompt, layer, token_index, feature_set_ids, set_mode, set_coefficient, set_target],
        outputs=[set_feature_table, set_metrics, set_target_table, set_feature_tsv, set_target_tsv],
    )
    set_sweep_btn.click(
        run_feature_set_sweep,
        inputs=[prompt, layer, token_index, set_sweep_target],
        outputs=[set_sweep_table, set_sweep_plot, set_sweep_note, set_sweep_tsv],
    )
    interaction_btn.click(
        run_feature_interaction,
        inputs=[prompt, layer, token_index, feature_set_ids, interaction_target],
        outputs=[interaction_table, interaction_metrics, interaction_plot, interaction_tsv],
    )
    geometry_btn.click(
        run_feature_geometry,
        inputs=[prompt, layer, token_index, feature_set_ids],
        outputs=[geometry_metrics, geometry_table, geometry_plot, geometry_tsv],
    )
    trace_btn.click(
        run_feature_trace,
        inputs=[prompt, contrast_layer, contrast_feature_id],
        outputs=[trace_metrics, trace_table, trace_plot, trace_tsv],
    )
    contrast_btn.click(
        run_concept_contrast,
        inputs=[contrast_feature_id, contrast_layer, contrast_n],
        outputs=[contrast_metrics, contrast_table, contrast_plot, contrast_tsv],
    )
    discovery_btn.click(
        run_concept_feature_discovery,
        inputs=[
            discovery_concept, discovery_layer, discovery_n, discovery_top_n, discovery_ranking,
            prompt, token_index,
        ],
        outputs=[
            discovery_metrics,
            discovery_table,
            discovery_plot,
            discovery_candidate,
            candidate_screen_ids,
            discovery_tsv,
        ],
    )
    candidate_screen_btn.click(
        run_candidate_causal_screen,
        inputs=[
            prompt,
            discovery_layer,
            token_index,
            candidate_screen_ids,
            candidate_screen_target,
            discovery_table,
        ],
        outputs=[
            candidate_screen_metrics,
            candidate_screen_table,
            candidate_screen_plot,
            discovery_candidate,
            candidate_screen_tsv,
            candidate_alignment_metrics,
            candidate_alignment_table,
            candidate_alignment_plot,
            candidate_alignment_tsv,
            candidate_specificity_ids,
        ],
    )
    candidate_specificity_btn.click(
        run_candidate_specificity_screen,
        inputs=[
            prompt,
            discovery_layer,
            token_index,
            candidate_specificity_ids,
            candidate_specificity_target,
            discovery_table,
        ],
        outputs=[
            candidate_specificity_metrics,
            candidate_specificity_table,
            candidate_specificity_plot,
            discovery_candidate,
            candidate_specificity_tsv,
            controlled_pattern_metrics,
            controlled_pattern_table,
            controlled_pattern_tsv,
            controlled_alignment_metrics,
            controlled_alignment_table,
            controlled_alignment_plot,
            controlled_alignment_tsv,
            cross_target_ids,
        ],
    )
    cross_target_btn.click(
        run_candidate_cross_target_profile,
        inputs=[
            prompt,
            discovery_layer,
            token_index,
            cross_target_ids,
            cross_target_text,
        ],
        outputs=[
            cross_target_metrics,
            cross_target_table,
            cross_target_plot,
            cross_target_summary_table,
            cross_target_pairwise_table,
            cross_target_pairwise_plot,
            cross_target_tsv,
            cross_target_summary_tsv,
            cross_target_pairwise_tsv,
        ],
    )
    candidate_specificity_table.select(
        select_candidate_row,
        inputs=[candidate_specificity_table],
        outputs=[discovery_candidate],
        queue=False,
    )
    candidate_screen_table.select(
        select_candidate_row,
        inputs=[candidate_screen_table],
        outputs=[discovery_candidate],
        queue=False,
    )
    discovery_table.select(
        select_candidate_row,
        inputs=[discovery_table],
        outputs=[discovery_candidate],
        queue=False,
    )
    use_candidate_btn.click(
        use_candidate_feature,
        inputs=[discovery_candidate],
        outputs=[feature_id, dose_feature_id, contrastive_feature_id, contrast_feature_id, candidate_use_status],
        queue=False,
    )
    cue_btn.click(
        run_feature_cue_scan,
        inputs=[contrast_feature_id, contrast_layer, cue_stem, cue_text],
        outputs=[cue_metrics, cue_table, cue_plot, cue_tsv],
    )
    cue_context_btn.click(
        run_feature_cue_context_scan,
        inputs=[contrast_feature_id, contrast_layer, cue_context_stems, cue_context_cues],
        outputs=[cue_context_metrics, cue_context_table, cue_context_plot, cue_context_tsv],
    )
    para_btn.click(
        run_paraphrase_compare,
        inputs=[para_a, para_b, para_layer, para_idx_a, para_idx_b, para_top_n],
        outputs=[para_tokens_a, para_tokens_b, para_metrics, para_table, para_plot, para_tsv],
    )
    trajectory_btn.click(
        run_layer_sweep,
        inputs=[trajectory_prompt, trajectory_token],
        outputs=[trajectory_tokens, trajectory_table, trajectory_plot, trajectory_tsv],
    )

    for button, source in [
        (feature_copy, feature_tsv),
        (token_prob_copy, token_prob_tsv),
        (target_token_copy, target_token_tsv),
        (dose_copy, dose_tsv),
        (contrastive_copy, contrastive_tsv),
        (set_feature_copy, set_feature_tsv),
        (set_target_copy, set_target_tsv),
        (set_sweep_copy, set_sweep_tsv),
        (interaction_copy, interaction_tsv),
        (geometry_copy, geometry_tsv),
        (trace_copy, trace_tsv),
        (contrast_copy, contrast_tsv),
        (discovery_copy, discovery_tsv),
        (candidate_screen_copy, candidate_screen_tsv),
        (candidate_alignment_copy, candidate_alignment_tsv),
        (candidate_specificity_copy, candidate_specificity_tsv),
        (controlled_pattern_copy, controlled_pattern_tsv),
        (controlled_alignment_copy, controlled_alignment_tsv),
        (cross_target_copy, cross_target_tsv),
        (cross_target_summary_copy, cross_target_summary_tsv),
        (cross_target_pairwise_copy, cross_target_pairwise_tsv),
        (cue_copy, cue_tsv),
        (cue_context_copy, cue_context_tsv),
        (para_copy, para_tsv),
        (trajectory_copy, trajectory_tsv),
    ]:
        _bind_copy(button, source)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        css=CSS,
        theme=THEME,
        ssr_mode=False,
        show_error=True,
    )
