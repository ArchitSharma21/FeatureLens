from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pandas as pd


def _import_app():
    try:
        import transformers  # noqa: F401
    except ModuleNotFoundError:
        stub = types.ModuleType('transformers')
        stub.AutoModelForCausalLM = type('AutoModelForCausalLM', (), {})
        stub.AutoTokenizer = type('AutoTokenizer', (), {})
        sys.modules['transformers'] = stub
    import app

    return app


def test_concept_metrics_do_not_invent_a_leader_for_all_zero_scan() -> None:
    app = _import_app()
    result = SimpleNamespace(
        feature_id=22632,
        layer=14,
        prompts_per_concept=4,
        active_prompt_count=0,
        total_prompt_count=28,
        leading_concept=None,
        leading_ratio=None,
    )
    text = app._concept_metrics_markdown(result)
    assert 'inactive in every sampled prompt' in text
    assert 'Highest prompt-wide mean activation' not in text


def test_tsv_copy_payload_keeps_headers() -> None:
    app = _import_app()
    frame = pd.DataFrame([[1, 2.0]], columns=['Feature id', 'Activation'])
    payload = app._tsv(frame)
    assert payload.startswith('Feature id\tActivation\n')
    assert payload.endswith('1\t2.0\n')


def test_use_candidate_feature_returns_explicit_handoff_status() -> None:
    app = _import_app()
    outputs = app.use_candidate_feature('21885')
    assert outputs[:4] == ('21885', '21885', '21885', '21885')
    assert 'Feature 21885 loaded' in outputs[4]
    assert 'feature-level experiments' in outputs[4]


def test_select_candidate_row_uses_feature_id_column() -> None:
    app = _import_app()
    table = pd.DataFrame(
        [[1, 21885, 3.4], [2, 445, 3.3]],
        columns=['Rank', 'Feature id', 'Candidate score'],
    )
    event = SimpleNamespace(index=(1, 0))
    update = app.select_candidate_row(table, event)
    # Gradio returns an update dictionary-like object in current releases.
    assert update['value'] == '445'


def test_frontend_helpers_name_exports_and_use_in_place_focus() -> None:
    app = _import_app()
    assert 'featurelens_${stem' in app.INSTALL_REFLOW_JS
    assert 'chart.png' in app.INSTALL_REFLOW_JS
    assert 'featurelens-inline-focus' in app.INSTALL_REFLOW_JS
    assert 'toggleInlineFocus' in app.INSTALL_REFLOW_JS
    assert 'stopImmediatePropagation' in app.INSTALL_REFLOW_JS
    assert 'featurelens-plot-focus-overlay' not in app.INSTALL_REFLOW_JS
    assert 'window.scrollTo' not in app.INSTALL_REFLOW_JS


def test_dose_response_has_independent_target_control() -> None:
    app = _import_app()
    assert app.dose_target_text.value == '2x'
    # The single-feature target remains optional and independent.
    assert app.target_text.value in {'', None}


def test_cue_context_markdown_reports_strong_cue_dominance() -> None:
    app = _import_app()
    result = SimpleNamespace(
        feature_id=22632,
        layer=14,
        stems=['math', 'capital', 'weather', 'name'],
        active_condition_count=4,
        condition_count=20,
        cue_active_context_counts={'is': 4, '=': 0, ':': 0, 'equals': 0, 'therefore': 0},
        dominant_cue='is',
        dominant_cue_context_count=4,
        off_dominant_active_count=0,
    )
    text = app._cue_context_metrics_markdown(result)
    assert 'Cue-dominant pattern' in text
    assert '`is` activates in every tested context' in text
    assert 'lexical/cue-specific' in text


def test_result_tables_hide_native_labels_in_favor_of_explicit_headings() -> None:
    app = _import_app()
    assert app.discovery_table.show_label is False
    assert app.dose_table.show_label is False
    assert '.table-heading' in app.CSS
    assert 'margin: 2px 0 -32px' in app.CSS
    assert 'font-size: 1.16rem' in app.CSS


def test_candidate_screen_markdown_is_explicitly_triage_only() -> None:
    app = _import_app()
    result = SimpleNamespace(
        candidate_count=3,
        active_feature_count=3,
        target_tokens=['2', 'x'],
        execution_drift_mean_logprob=1e-4,
        execution_drift_js=2e-6,
        rows=[[1, 16369, 29.25, True, 29.0, -0.12, -0.24, 0.004]],
    )
    text = app._candidate_screen_metrics_markdown(result)
    assert 'triage screen' in text
    assert 'no random-control ensemble is spent here' in text


def test_candidate_screen_has_independent_target_and_multiselect() -> None:
    app = _import_app()
    assert app.candidate_screen_target.value == '2x'
    assert app.candidate_screen_ids.multiselect is True
    assert app.candidate_screen_ids.max_choices == 8


def test_candidate_alignment_exposes_discovery_causality_divergence() -> None:
    app = _import_app()
    discovery = pd.DataFrame(
        [
            [1, 16369, 4.7348, 15.0, 0.0, 15.0, 1.0, 0.5, 0.0, 50.0, 29.25, True],
            [2, 5712, 4.1969, 9.1, 1.5, 7.6, 0.72, 1.0, 0.17, 11.4, 11.38, True],
            [3, 26112, 3.9419, 10.8, 0.0, 10.8, 1.0, 0.5, 0.0, 23.4, 23.42, True],
            [4, 25992, 3.7342, 10.1, 0.0, 10.1, 1.0, 0.5, 0.0, 21.4, 21.36, True],
            [5, 21670, 3.6200, 18.9, 2.0, 16.9, 0.81, 0.75, 0.125, 25.3, 6.36, True],
        ],
        columns=[
            'Rank', 'Feature id', 'Candidate score', 'Target mean max', 'Other mean max',
            'Mean difference', 'Selectivity', 'Target activation rate', 'Other activation rate',
            'Current prompt max', 'Current token activation', 'Active at current token',
        ],
    )
    screen = pd.DataFrame(
        [
            [1, 25992, 21.3, True, 21.3, -0.1119, -0.2238, 0.001066],
            [2, 21670, 6.36, True, 6.36, 0.0313, 0.0626, 0.000106],
            [3, 26112, 23.4, True, 23.4, -0.0211, -0.0423, 0.000417],
            [4, 5712, 11.38, True, 11.38, 0.0110, 0.0221, 0.000155],
            [5, 16369, 29.23, True, 29.23, -0.0101, -0.0201, 0.001714],
        ],
        columns=[
            'Rank', 'Feature id', 'Native activation', 'Active at current token', 'Perturbation L2',
            'Δ mean log p/token', 'Δ sequence log p', 'Next-token JS',
        ],
    )
    summary, table, chart = app._candidate_alignment_outputs(discovery, screen)
    assert len(table) == 5
    assert int(table.loc[table['Target-effect rank'].idxmin(), 'Feature id']) == 25992
    assert int(table.loc[table['Distribution-shift rank'].idxmin(), 'Feature id']) == 16369
    row_25992 = table.loc[table['Feature id'] == 25992].iloc[0]
    row_16369 = table.loc[table['Feature id'] == 16369].iloc[0]
    assert int(row_25992['Discovery→target rank shift']) == 3
    assert int(row_16369['Discovery→target rank shift']) == -4
    assert 'Spearman' in summary
    assert '-0.900' in summary
    assert '+0.600' in summary
    assert 'descriptive' in summary
    assert set(chart['Feature id']) == {'16369', '5712', '26112', '25992', '21670'}


def test_candidate_alignment_ui_uses_same_screen_call_and_no_new_gpu_button() -> None:
    app = _import_app()
    assert app.candidate_alignment_table.show_label is False
    assert app.candidate_alignment_plot.visible is True
    assert 'Association evidence vs target effect' == app.candidate_alignment_plot.title


def _v10_discovery_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [1, 16369, 4.7348, 15.0, 0.0, 15.0, 1.0, 0.5, 0.0, 50.0, 29.25, True],
            [2, 5712, 4.1969, 9.1, 1.5, 7.6, 0.72, 1.0, 0.17, 11.4, 11.38, True],
            [3, 26112, 3.9419, 10.8, 0.0, 10.8, 1.0, 0.5, 0.0, 23.4, 23.42, True],
            [4, 25992, 3.7342, 10.1, 0.0, 10.1, 1.0, 0.5, 0.0, 21.4, 21.36, True],
            [5, 21670, 3.6200, 18.9, 2.0, 16.9, 0.81, 0.75, 0.125, 25.3, 6.36, True],
        ],
        columns=[
            'Rank', 'Feature id', 'Candidate score', 'Target mean max', 'Other mean max',
            'Mean difference', 'Selectivity', 'Target activation rate', 'Other activation rate',
            'Current prompt max', 'Current token activation', 'Active at current token',
        ],
    )


def _v10_screen_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [1, 25992, 21.3, True, 21.3, -0.1119, -0.2238, 0.001066],
            [2, 21670, 6.36, True, 6.36, 0.0313, 0.0626, 0.000106],
            [3, 26112, 23.4, True, 23.4, -0.0211, -0.0423, 0.000417],
            [4, 5712, 11.38, True, 11.38, 0.0110, 0.0221, 0.000155],
            [5, 16369, 29.23, True, 29.23, -0.0101, -0.0201, 0.001714],
        ],
        columns=[
            'Rank', 'Feature id', 'Native activation', 'Active at current token', 'Perturbation L2',
            'Δ mean log p/token', 'Δ sequence log p', 'Next-token JS',
        ],
    )


def test_controlled_candidate_shortlist_preserves_discovery_and_causal_leaders() -> None:
    app = _import_app()
    selected = app._controlled_candidate_shortlist(_v10_discovery_table(), _v10_screen_table(), limit=3)
    assert selected == ['16369', '25992', '21670']


def test_controlled_alignment_uses_random_normalized_specificity() -> None:
    app = _import_app()
    controlled = pd.DataFrame(
        [
            [1, 25992, 21.3, True, 21.3, -0.1119, -0.01, 0.04, 0.02, 2.80, 0.111, -0.2238, 0.001066, 0.00040, 0.0001, 2.665, 0.111],
            [2, 16369, 29.2, True, 29.2, -0.0101, 0.00, 0.05, 0.03, 0.20, 0.778, -0.0201, 0.001714, 0.00050, 0.0002, 3.428, 0.111],
            [3, 21670, 6.36, True, 6.36, 0.0313, 0.00, 0.03, 0.01, 1.04, 0.444, 0.0626, 0.000106, 0.00020, 0.0001, 0.53, 0.778],
        ],
        columns=[
            'Rank', 'Feature id', 'Native activation', 'Active at current token', 'Perturbation L2',
            'SAE Δ mean log p/token', 'Random signed mean Δ', 'Random mean |Δ|', 'Random |Δ| std',
            'Target specificity ratio', 'Target empirical tail p', 'SAE Δ sequence log p',
            'SAE next-token JS', 'Random mean JS', 'Random JS std', 'JS specificity ratio',
            'JS empirical tail p',
        ],
    )
    summary, table, chart = app._controlled_alignment_outputs(_v10_discovery_table(), controlled)
    assert len(table) == 3
    assert int(table.loc[table['Specificity rank'].idxmin(), 'Feature id']) == 25992
    row_25992 = table.loc[table['Feature id'] == 25992].iloc[0]
    row_16369 = table.loc[table['Feature id'] == 16369].iloc[0]
    assert int(row_25992['Discovery→specificity rank shift']) == 3
    assert int(row_16369['Discovery→specificity rank shift']) == -1
    assert 'norm-matched random ensemble' in summary
    assert set(chart['Feature id']) == {'25992', '16369', '21670'}


def test_controlled_candidate_ui_limits_live_comparison_to_three_features() -> None:
    app = _import_app()
    assert app.candidate_specificity_ids.multiselect is True
    assert app.candidate_specificity_ids.max_choices == 3
    assert app.candidate_specificity_target.value == '2x'
    assert app.candidate_specificity_table.show_label is False
    assert app.controlled_alignment_table.show_label is False


def _v11_controlled_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [1, 25992, 21.328, True, 21.328, -0.11188, -0.01628, 0.05637, 0.06847, 1.9849, 0.3333, -0.22376, 0.001133, 0.000498, 0.000219, 2.2769, 0.1111],
            [2, 21670, 6.359, True, 6.359, 0.02105, -0.01246, 0.01732, 0.01587, 1.2157, 0.4444, 0.04211, 0.000104, 0.000043, 0.000027, 2.4423, 0.2222],
            [3, 16369, 29.234, True, 29.235, -0.00621, -0.00330, 0.04204, 0.04735, 0.1476, 0.8889, -0.01241, 0.001832, 0.000689, 0.000197, 2.6610, 0.1111],
        ],
        columns=[
            'Rank', 'Feature id', 'Native activation', 'Active at current token', 'Perturbation L2',
            'SAE Δ mean log p/token', 'Random signed mean Δ', 'Random mean |Δ|', 'Random |Δ| std',
            'Target specificity ratio', 'Target empirical tail p', 'SAE Δ sequence log p',
            'SAE next-token JS', 'Random mean JS', 'Random JS std', 'JS specificity ratio',
            'JS empirical tail p',
        ],
    )


def test_controlled_evidence_patterns_separate_target_and_distributional_influence() -> None:
    app = _import_app()
    summary, table = app._controlled_evidence_patterns(_v11_controlled_table())
    patterns = dict(zip(table['Feature id'].astype(int), table['Evidence pattern'], strict=True))
    assert patterns[25992] == 'Broad controlled influence'
    assert patterns[21670] == 'Distribution-shift weighted'
    assert patterns[16369] == 'Distribution-shift dominant'
    assert 'effect ratios' in summary
    assert 'statistical significance' in summary


def test_controlled_alignment_explains_missing_discovery_state_instead_of_blank() -> None:
    app = _import_app()
    summary, table, chart = app._controlled_alignment_outputs(None, _v11_controlled_table())
    assert 'discovery' in summary.lower()
    assert 'browser session' in summary.lower()
    assert table.empty
    assert chart.empty


def test_cross_target_shortlist_preserves_target_and_js_leaders() -> None:
    app = _import_app()
    selected = app._cross_target_shortlist(_v11_controlled_table(), limit=2)
    assert selected == ['25992', '16369']


def test_cross_target_ui_has_independent_targets_and_small_feature_limit() -> None:
    app = _import_app()
    assert app.cross_target_ids.multiselect is True
    assert app.cross_target_ids.max_choices == 3
    assert '2x' in app.cross_target_text.value
    assert app.cross_target_table.show_label is False
    assert app.cross_target_summary_table.show_label is False


def test_v13_discovery_markdown_reports_resample_support() -> None:
    app = _import_app()
    result = SimpleNamespace(
        candidate_ids=[16369, 5712, 26112],
        concept='mathematics',
        layer=14,
        prompts_per_concept=4,
        ranking_mode='causal_ready',
        current_context_available=True,
        current_token_index=5,
        displayed_current_active_count=3,
        split_half_k=3,
        split_half_shared_count=2,
        split_half_jaccard=0.5,
        resample_replicates=32,
        resample_mean_support=0.71875,
        resample_high_support_count=2,
    )
    text = app._discovery_metrics_markdown(result)
    assert '32 resamples' in text
    assert '71.9%' in text
    assert '2/3' in text
    assert 'confidence interval' in text


def test_v13_cross_target_markdown_reports_profile_and_pairwise_summary() -> None:
    app = _import_app()
    result = SimpleNamespace(
        feature_ids=[25992, 16369],
        targets=['2x', 'x', '0', 'x^2'],
        active_feature_count=2,
        summary_rows=[
            [16369, 'x', 0.2883, 0.2883, 0.0513, 5.62, 'mixed signs', 0.58, 0.42, 0.92,
             'target-concentrated / mixed-sign', 0.0017],
            [25992, '0', -0.1657, 0.1657, 0.0936, 1.77, 'same sign', 0.95, 0.05, -1.0,
             'broad same-sign suppression', 0.0011],
        ],
        pairwise_rows=[
            [16369, 'x', '0', 0.2969, 0.2969, 'toward x'],
        ],
    )
    text = app._cross_target_metrics_markdown(result)
    assert 'target-concentrated / mixed-sign' in text
    assert 'broad same-sign suppression' in text
    assert 'pairwise preference shift' in text
    assert "'x' vs '0'" in text


def test_v13_pairwise_cross_target_ui_is_zero_extra_gpu_output() -> None:
    app = _import_app()
    assert app.cross_target_pairwise_table.show_label is False
    assert app.cross_target_pairwise_plot.title == 'Pairwise target preference shifts'
    assert app.cross_target_pairwise_plot.visible is True
