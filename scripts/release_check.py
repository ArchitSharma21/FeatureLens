from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_SIZE_BYTES = 5_000_000

REQUIRED = [
    "README.md",
    "DESIGN.md",
    "app.py",
    "requirements.txt",
    "research_config.json",
    "featurelens/runtime.py",
    "featurelens/sae.py",
    "featurelens/interventions.py",
    "featurelens/stats.py",
    "featurelens/study.py",
    "experiments/run_all.py",
    "experiments/run_causal.py",
    "experiments/run_feature_sets.py",
    "experiments/analyze_stability.py",
    "experiments/analyze_study.py",
    "experiments/run_analysis_only.py",
    "data/prompts.jsonl",
    "data/causal_tasks.jsonl",
    "notebooks/README.md",
    "notebooks/FeatureLens_Offline_Study_Colab.ipynb",
    "notebooks/FeatureLens_Causal_Addendum_Colab.ipynb",
    "scripts/ui_smoke.py",
    "scripts/validate_artifacts.py",
    "artifacts/feature_catalog.csv",
    "artifacts/layer_metrics.csv",
    "artifacts/stability.csv",
    "artifacts/selection_stability.csv",
    "artifacts/causal_results_final_token.csv",
    "artifacts/causal_results_max_active.csv",
    "artifacts/causal_position_summary.csv",
    "artifacts/feature_set_results.csv",
    "artifacts/study_feature_summary.csv",
    "artifacts/study_summary.json",
    "artifacts/summary.json",
    "artifacts/report.md",
]


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def repository_candidates() -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit("Git is required to run the FeatureLens release check.") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Could not inspect repository files with Git: {exc.stderr.strip()}"
        ) from exc

    return [
        ROOT / rel
        for rel in result.stdout.splitlines()
        if rel.strip() and (ROOT / rel.strip()).is_file()
    ]


def check_required_files() -> None:
    missing = [name for name in REQUIRED if not (ROOT / name).exists()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")


def check_config(config: dict) -> None:
    expected = {
        "model_id": "Qwen/Qwen3-1.7B-Base",
        "layers": [4, 14, 26],
        "sae_width": 32768,
        "discovery_prompts": 224,
        "causal_tasks": 28,
        "feature_set_sizes": [1, 3, 5],
        "live_random_controls": 8,
        "offline_random_controls_default": 8,
        "offline_selection_resamples": 128,
        "offline_causal_position_policies": [
            "final_token",
            "max_feature_activation",
        ],
        "primary_offline_causal_position_policy": "max_feature_activation",
        "release_status": "final",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise SystemExit(
                f"Unexpected {key}: {config.get(key)!r}. Expected {value!r}."
            )

    if "prompt-wide" not in str(config.get("offline_feature_pooling", "")):
        raise SystemExit("Offline SAE concept evidence must use prompt-wide pooling.")

    concepts = config.get("concepts", [])
    if "german_language" not in concepts or "french_language" in concepts:
        raise SystemExit("Controlled language concept must be german_language.")


def check_datasets(config: dict) -> tuple[list[dict], list[dict]]:
    prompts = load_jsonl(ROOT / "data" / "prompts.jsonl")
    causal = load_jsonl(ROOT / "data" / "causal_tasks.jsonl")

    if len(prompts) != config["discovery_prompts"]:
        raise SystemExit(
            f"Discovery prompt count mismatch: {len(prompts)} != "
            f"{config['discovery_prompts']}."
        )
    if len(causal) != config["causal_tasks"]:
        raise SystemExit(
            f"Causal task count mismatch: {len(causal)} != {config['causal_tasks']}."
        )

    concept_counts = Counter(row["concept"] for row in prompts)
    if set(concept_counts) != set(config["concepts"]):
        raise SystemExit("Discovery concepts do not match research_config.json.")
    if len(set(concept_counts.values())) != 1:
        raise SystemExit(f"Discovery concepts are not balanced: {dict(concept_counts)}")

    pair_counts = Counter(row["pair_id"] for row in prompts)
    if set(pair_counts.values()) != {2}:
        raise SystemExit("Every discovery paraphrase pair must contain exactly two prompts.")

    return prompts, causal


def check_study_summary() -> None:
    study = json.loads(
        (ROOT / "artifacts" / "study_summary.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (ROOT / "artifacts" / "summary.json").read_text(encoding="utf-8")
    )

    if study.get("primary_causal_position_policy") != "max_feature_activation":
        raise SystemExit("Committed study must use max_feature_activation as primary policy.")
    if "causal task" not in str(study.get("causal_statistical_unit", "")).lower():
        raise SystemExit("Committed study must document causal-task-level inference.")
    if float(study.get("max_active_feature_coverage", 0.0)) <= float(
        study.get("final_token_feature_coverage", 0.0)
    ):
        raise SystemExit("Expected max-active coverage to exceed final-token coverage.")
    if float(study.get("max_active_target_specificity_ratio", 0.0)) <= 1.0:
        raise SystemExit("Committed max-active study specificity ratio is invalid.")

    headline = str(summary.get("headline", ""))
    if "0.962" not in headline or "2.33" not in headline:
        raise SystemExit("Committed summary.json does not contain the finalized measured headline.")


def check_readme() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    required = [
        "0.962 held-out AUROC",
        "2.33×",
        "28.6%",
        "82.1%",
        "notebooks/FeatureLens_Offline_Study_Colab.ipynb",
        "artifacts/report.md",
    ]
    missing = [text for text in required if text not in readme]
    if missing:
        raise SystemExit(f"README.md missing finalized study content: {missing}")


def check_oversized_files() -> None:
    oversized: list[str] = []
    for path in repository_candidates():
        size = path.stat().st_size
        if size > MAX_FILE_SIZE_BYTES:
            oversized.append(
                f"{path.relative_to(ROOT)} ({size / 1_000_000:.1f} MB)"
            )

    if oversized:
        formatted = "\n  - ".join(oversized)
        raise SystemExit(
            "Repository contains unexpectedly large tracked/unignored candidates:\n"
            f"  - {formatted}\n\n"
            "Model weights, SAE checkpoints, activation dumps, virtual environments, "
            "and caches should not be committed."
        )


def main() -> None:
    check_required_files()

    config = json.loads((ROOT / "research_config.json").read_text(encoding="utf-8"))
    check_config(config)
    prompts, causal = check_datasets(config)
    check_study_summary()
    check_readme()
    check_oversized_files()

    print("FeatureLens release check: PASS")
    print(f"  discovery prompts: {len(prompts)}")
    print(f"  causal tasks: {len(causal)}")
    print(f"  layers: {config['layers']}")
    print("  committed offline study: complete")
    print("  release: 1.0.0")


if __name__ == "__main__":
    main()
