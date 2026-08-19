from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_SIZE_BYTES = 5_000_000  # 5 MB

REQUIRED = [
    'README.md',
    'DESIGN.md',
    'app.py',
    'requirements.txt',
    'research_config.json',
    'featurelens/runtime.py',
    'featurelens/sae.py',
    'featurelens/interventions.py',
    'featurelens/metrics.py',
    'featurelens/stats.py',
    'featurelens/study.py',
    'experiments/run_all.py',
    'experiments/run_causal.py',
    'experiments/run_causal_addendum.py',
    'experiments/run_feature_sets.py',
    'experiments/analyze_stability.py',
    'experiments/analyze_study.py',
    'experiments/run_analysis_only.py',
    'data/prompts.jsonl',
    'data/causal_tasks.jsonl',
    'docs/VALIDATION.md',
    'docs/OFFLINE_STUDY.md',
    'docs/COLAB.md',
    'docs/CAUSAL_ADDENDUM.md',
    'notebooks/FeatureLens_Offline_Study_Colab.ipynb',
    'notebooks/FeatureLens_Causal_Addendum_Colab.ipynb',
    'scripts/ui_smoke.py',
    'tests/test_offline_study.py',
    'scripts/validate_artifacts.py',
]


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def repository_candidates() -> list[Path]:
    """Return tracked files plus untracked files that are not ignored by Git."""
    try:
        result = subprocess.run(
            ['git', 'ls-files', '--cached', '--others', '--exclude-standard'],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit('Git is required to run the FeatureLens release check.') from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f'Could not inspect repository files with Git: {exc.stderr.strip()}'
        ) from exc

    paths: list[Path] = []
    for relative_path in result.stdout.splitlines():
        relative_path = relative_path.strip()
        if not relative_path:
            continue
        path = ROOT / relative_path
        if path.is_file():
            paths.append(path)
    return paths


def check_required_files() -> None:
    missing = [path for path in REQUIRED if not (ROOT / path).exists()]
    if missing:
        raise SystemExit(f'Missing required files: {missing}')


def check_config(config: dict) -> None:
    expected = {
        'layers': [4, 14, 26],
        'model_id': 'Qwen/Qwen3-1.7B-Base',
        'sae_width': 32768,
        'dose_response_multipliers': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
        'feature_set_sizes': [1, 3, 5],
        'live_random_controls': 8,
        'offline_random_controls_default': 8,
        'concept_contrast_prompts_per_concept': 4,
        'interaction_feature_limit': 5,
        'live_geometry_feature_limit': 8,
        'concept_contrast_pooling': 'max activation across non-padding prompt tokens',
        'candidate_causal_screen_limit': 8,
        'candidate_specificity_limit': 3,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise SystemExit(f'Unexpected {key}: {config.get(key)!r}. Expected {value!r}.')

    required_live_v04 = {
        'batch_context_null_reference',
        'random_control_ensemble',
        'individual_vs_joint_interaction_decomposition',
        'promptwide_paraphrase_robustness',
        'controlled_concept_contrast_scan',
        'copy_tables_with_headers',
    }
    actual_live_v04 = set(config.get('live_features_v0_4', []))
    if actual_live_v04 != required_live_v04:
        raise SystemExit(
            'research_config.json live_features_v0_4 mismatch: '
            f'{sorted(actual_live_v04)}'
        )

    required_live_v05 = {
        'wide_centered_responsive_layout',
        'copy_feedback',
        'dynamic_height_reflow_observer',
        'promptwide_concept_contrast_scan',
        'feature_token_activation_trace',
        'contrastive_continuation_preference_test',
        'feature_decoder_geometry',
    }
    actual_live_v05 = set(config.get('live_features_v0_5', []))
    if actual_live_v05 != required_live_v05:
        raise SystemExit(
            'research_config.json live_features_v0_5 mismatch: '
            f'{sorted(actual_live_v05)}'
        )

    required_live_v06 = {
        'start_here_plain_language_onboarding',
        'persistent_workbench_context_banner',
        'explicit_per_experiment_feature_selectors',
        'plot_fullscreen_and_export_controls',
        'consistent_heading_and_table_typography',
        'concept_guided_candidate_feature_discovery',
        'completion_cue_sensitivity_scan',
    }
    actual_live_v06 = set(config.get('live_features_v0_6', []))
    if actual_live_v06 != required_live_v06:
        raise SystemExit(
            'research_config.json live_features_v0_6 mismatch: '
            f'{sorted(actual_live_v06)}'
        )

    required_live_v07 = {
        'cleaned_nonaccordion_experiment_layout',
        'focused_fullscreen_modal_for_tables_and_plots',
        'descriptive_plot_export_filenames',
        'german_language_control_concept',
        'balanced_candidate_ranking_and_current_prompt_compatibility',
        'click_to_select_candidate_rows',
        'completion_cue_context_matrix',
    }
    actual_live_v07 = set(config.get('live_features_v0_7', []))
    if actual_live_v07 != required_live_v07:
        raise SystemExit(
            'research_config.json live_features_v0_7 mismatch: '
            f'{sorted(actual_live_v07)}'
        )

    required_live_v08 = {
        'bounded_plot_focus_overlay_with_scroll_restore',
        'explicit_result_table_headings',
        'standalone_dose_response_target_and_feature_inputs',
        'causal_ready_current_token_candidate_ranking',
        'cue_dominance_specificity_interpretation',
        'muted_cue_context_plot_palette',
    }
    actual_live_v08 = set(config.get('live_features_v0_8', []))
    if actual_live_v08 != required_live_v08:
        raise SystemExit(
            'research_config.json live_features_v0_8 mismatch: ' f'{sorted(actual_live_v08)}'
        )

    required_live_v09 = {
        'in_place_aspect_preserving_plot_and_table_focus',
        'compact_table_heading_alignment',
        'concise_independent_dose_response_copy',
        'batched_candidate_causal_triage',
        'gpu_budget_aware_hf_validation_scope',
    }
    actual_live_v09 = set(config.get('live_features_v0_9', []))
    if actual_live_v09 != required_live_v09:
        raise SystemExit(
            'research_config.json live_features_v0_9 mismatch: ' f'{sorted(actual_live_v09)}'
        )

    required_live_v10 = {
        'discovery_to_causality_alignment_table',
        'association_evidence_vs_target_effect_scatter',
        'descriptive_spearman_concordance_summary',
        'target_effect_vs_distribution_shift_rank_separation',
        'no_extra_gpu_candidate_synthesis',
    }
    actual_live_v10 = set(config.get('live_features_v0_10', []))
    if actual_live_v10 != required_live_v10:
        raise SystemExit(
            'research_config.json live_features_v0_10 mismatch: ' f'{sorted(actual_live_v10)}'
        )

    required_live_v11 = {
        'controlled_multi_candidate_random_specificity_screen',
        'strategic_discovery_target_js_shortlist',
        'association_vs_controlled_causality_alignment',
        'target_specificity_vs_js_specificity_separation',
        'single_new_gpu_call_hf_acceptance',
    }
    actual_live_v11 = set(config.get('live_features_v0_11', []))
    if actual_live_v11 != required_live_v11:
        raise SystemExit(
            'research_config.json live_features_v0_11 mismatch: ' f'{sorted(actual_live_v11)}'
        )

    required_live_v12 = {
        'controlled_evidence_pattern_synthesis',
        'split_half_discovery_stability',
        'cross_target_candidate_profile',
        'missing_discovery_alignment_fallback',
        'gpu_budget_aware_touched_path_validation',
    }
    actual_live_v12 = set(config.get('live_features_v0_12', []))
    if actual_live_v12 != required_live_v12:
        raise SystemExit(
            'research_config.json live_features_v0_12 mismatch: ' f'{sorted(actual_live_v12)}'
        )

    required_live_v13 = {
        'balanced_bootstrap_candidate_support',
        'cross_target_effect_concentration',
        'pairwise_target_preference_shifts',
        'zero_extra_gpu_evidence_synthesis',
        'touched_path_only_hf_validation',
    }
    actual_live_v13 = set(config.get('live_features_v0_13', []))
    if actual_live_v13 != required_live_v13:
        raise SystemExit(
            'research_config.json live_features_v0_13 mismatch: ' f'{sorted(actual_live_v13)}'
        )
    required_offline_v14 = {
        'promptwide_offline_sae_feature_pooling',
        'separate_final_token_sparse_activation_artifacts',
        'activation_resample_candidate_stability',
        'cross_concept_association_vs_random_normalized_causality',
        'offline_study_dashboard',
        'resume_safe_full_study_runner',
        'cpu_only_analysis_rerun',
        'offline_artifact_schema_validation',
    }
    actual_offline_v14 = set(config.get('offline_features_v0_14', []))
    if actual_offline_v14 != required_offline_v14:
        raise SystemExit(
            'research_config.json offline_features_v0_14 mismatch: '
            f'{sorted(actual_offline_v14)}'
        )

    required_v15 = {
        'project_design_contract',
        'flat_research_instrument_visual_system',
        'dual_typeface_hierarchy',
        'concise_data_first_result_copy',
        'muted_cross_target_chart_series',
        'colab_offline_runner_notebook',
        'task_level_causal_and_feature_set_resume',
    }
    actual_v15 = set(config.get('ui_and_runner_features_v0_15', []))
    if actual_v15 != required_v15:
        raise SystemExit(
            'research_config.json ui_and_runner_features_v0_15 mismatch: '
            f'{sorted(actual_v15)}'
        )

    required_v16 = {
        'final_token_vs_max_feature_activation_causal_position_sensitivity',
        'causal_task_level_statistical_inference',
        'coverage_separated_from_conditional_effect_strength',
        'exact_small_sample_sign_flip_tests',
        'causal_addendum_colab_runner',
        'position_sensitivity_study_dashboard',
    }
    actual_v16 = set(config.get('offline_features_v0_16', []))
    if actual_v16 != required_v16:
        raise SystemExit(
            'research_config.json offline_features_v0_16 mismatch: '
            f'{sorted(actual_v16)}'
        )
    if config.get('offline_causal_position_policies') != ['final_token', 'max_feature_activation']:
        raise SystemExit('Offline causal position policies must be final_token and max_feature_activation.')
    if config.get('primary_offline_causal_position_policy') != 'max_feature_activation':
        raise SystemExit('Primary offline causal position policy must be max_feature_activation.')
    if config.get('offline_selection_resamples') != 128:
        raise SystemExit('Offline selection resamples must be 128.')
    if 'prompt-wide' not in str(config.get('offline_feature_pooling', '')):
        raise SystemExit('Offline feature pooling must be prompt-wide.')

    if config.get('discovery_resample_replicates') != 32:
        raise SystemExit('Discovery live resample count must be 32.')
    if config.get('cross_target_feature_limit') != 3 or config.get('cross_target_target_limit') != 5:
        raise SystemExit('Cross-target live limits must be 3 features and 5 targets.')

    if 'german_language' not in config.get('concepts', []) or 'french_language' in config.get('concepts', []):
        raise SystemExit('research_config.json must use german_language and must not contain french_language.')


def check_datasets(config: dict) -> tuple[list[dict], list[dict]]:
    prompts = load_jsonl(ROOT / 'data' / 'prompts.jsonl')
    causal = load_jsonl(ROOT / 'data' / 'causal_tasks.jsonl')

    if len(prompts) != config.get('discovery_prompts'):
        raise SystemExit(
            f'Discovery prompt count mismatch: found {len(prompts)}, '
            f'expected {config.get("discovery_prompts")}.'
        )
    if len(causal) != config.get('causal_tasks'):
        raise SystemExit(
            f'Causal task count mismatch: found {len(causal)}, '
            f'expected {config.get("causal_tasks")}.'
        )

    concept_counts = Counter(row['concept'] for row in prompts)
    if set(concept_counts) != set(config.get('concepts', [])):
        raise SystemExit('Discovery dataset concepts do not match research_config.json.')
    if len(set(concept_counts.values())) != 1:
        raise SystemExit(f'Discovery concepts are not balanced: {dict(concept_counts)}')

    pair_counts = Counter(row['pair_id'] for row in prompts)
    if set(pair_counts.values()) != {2}:
        raise SystemExit('Every discovery paraphrase pair must contain exactly two prompts.')

    return prompts, causal


def check_oversized_files() -> None:
    oversized: list[str] = []
    for path in repository_candidates():
        size_bytes = path.stat().st_size
        if size_bytes > MAX_FILE_SIZE_BYTES:
            relative = path.relative_to(ROOT)
            oversized.append(f'{relative} ({size_bytes / 1_000_000:.1f} MB)')

    if oversized:
        formatted = '\n  - '.join(oversized)
        raise SystemExit(
            'Repository contains unexpectedly large tracked/unignored candidates:\n'
            f'  - {formatted}\n\n'
            'If a file is a legitimate local artifact, add it to .gitignore. Model weights, '
            'SAE checkpoints, activation dumps, virtual environments, and caches should not be committed.'
        )


def check_readme() -> None:
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    required_strings = [
        'sdk: gradio',
        'sdk_version: "6.24.0"',
        'Qwen/Qwen3-1.7B-Base',
        'Qwen-Scope',
        'random controls',
        'prompt-wide',
        'offline study',
        '-m experiments.run_all --resume',
        '--activation-batch-size',
        'validate_artifacts',
        'FeatureLens_Offline_Study_Colab.ipynb',
        'FeatureLens_Causal_Addendum_Colab.ipynb',
        'max-feature-activation',
        'causal task',
        'DESIGN.md',
    ]
    missing = [value for value in required_strings if value.lower() not in readme.lower()]
    if missing:
        raise SystemExit(f'README.md is missing required v0.16 content: {missing}')

    # Public README should not lead with release-train marketing. Version history belongs in CHANGELOG.
    if '> **v0.' in readme or '## v0.' in readme:
        raise SystemExit('README.md should not contain visible release-announcement/version-history sections.')


def check_pyproject() -> None:
    text = (ROOT / 'pyproject.toml').read_text(encoding='utf-8')
    if 'version = "0.16.0"' not in text:
        raise SystemExit('pyproject.toml must declare version 0.16.0.')


def main() -> None:
    check_required_files()
    config = json.loads((ROOT / 'research_config.json').read_text(encoding='utf-8'))
    check_config(config)
    prompts, causal = check_datasets(config)
    check_oversized_files()
    check_readme()
    check_pyproject()

    print('FeatureLens release check: PASS')
    print(f'  discovery prompts: {len(prompts)}')
    print(f'  causal tasks: {len(causal)}')
    print(f'  layers: {config["layers"]}')
    print(f'  feature-set sizes: {config["feature_set_sizes"]}')
    print(f'  random controls: {config["live_random_controls"]}')
    print('  release: v0.16.0')


if __name__ == '__main__':
    main()
