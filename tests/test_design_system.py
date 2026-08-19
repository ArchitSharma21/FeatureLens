from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / 'app.py').read_text(encoding='utf-8')
DESIGN = (ROOT / 'DESIGN.md').read_text(encoding='utf-8')


def test_design_contract_is_present_and_specific():
    assert 'research instrument' in DESIGN.lower()
    assert 'no gradients' in DESIGN.lower()
    assert 'nested-card' in DESIGN.lower() or 'nested cards' in DESIGN.lower()
    assert 'Segoe UI' in DESIGN
    assert 'Georgia' in DESIGN
    assert 'Copy TSV' in DESIGN


def test_public_ui_avoids_common_generated_frontend_tells():
    lowered = APP.lower()
    assert 'linear-gradient' not in lowered
    assert 'radial-gradient' not in lowered
    assert 'backdrop-filter' not in lowered
    assert 'gr.accordion(' not in lowered
    assert 'start here' not in lowered
    assert 'unlock your' not in lowered
    assert 'supercharge' not in lowered
    assert 'all-in-one' not in lowered
    assert '<h1>featurelens</h1>' in lowered
    assert '<h1>featurelens v' not in lowered


def test_ui_has_explicit_type_and_action_roles():
    assert '--fl-body: "Segoe UI"' in APP
    assert '--fl-display: Georgia' in APP
    assert '.action-btn' in APP
    assert '.copy-btn' in APP
    assert 'Copy TSV' in APP
    assert 'with gr.Tab("Features")' in APP


def test_cross_target_charts_use_restrained_fixed_series_palette():
    assert '"Feature A": INK_TEAL' in APP
    assert '"Feature B": INK_UMBER' in APP
    assert '"Feature C": INK_PLUM' in APP
