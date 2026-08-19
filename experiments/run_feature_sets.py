from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.common import ARTIFACT_DIR, DATA_DIR, load_jsonl, set_seed
from featurelens.config import SETTINGS
from featurelens.interventions import (
    InterventionSpec,
    joint_residual_delta,
    normalized_random_control,
)
from featurelens.metrics import js_divergence_from_logits, sequence_logprob_summary
from featurelens.sae import SAEStore
from featurelens.selection import load_feature_sets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run top-k joint SAE feature-set ablations.')
    parser.add_argument('--tasks', type=Path, default=DATA_DIR / 'causal_tasks.jsonl')
    parser.add_argument('--catalog', type=Path, default=ARTIFACT_DIR / 'feature_catalog.csv')
    parser.add_argument('--output', type=Path, default=ARTIFACT_DIR / 'feature_set_results.csv')
    parser.add_argument('--sizes', type=int, nargs='+', default=[1, 3, 5])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--random-controls', type=int, default=8)
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from task-level rows already checkpointed in --output.',
    )
    return parser.parse_args()




def _completion_marker(path: Path) -> Path:
    return path.with_suffix(path.suffix + '.complete')


def _write_rows_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_checkpoint_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def hidden_from_output(output):
    return output[0] if isinstance(output, tuple) else output


def replace_hidden(output, hidden):
    return (hidden, *output[1:]) if isinstance(output, tuple) else hidden


def _make_capture_hook(capture: dict):
    """Bind a per-task capture dictionary before registering the hook."""

    def capture_hook(_module, _inp, output):
        if 'hidden' not in capture:
            capture['hidden'] = hidden_from_output(output).detach()

    return capture_hook


def _make_batch_edit_hook(
    applied: dict[str, bool],
    prompt_len: int,
    deltas: torch.Tensor,
):
    """Bind per-task edit state so hooks cannot capture a later loop iteration."""

    def edit_hook(_module, _inp, output):
        if applied['done']:
            return output
        hidden = hidden_from_output(output)
        modified = hidden.clone()
        modified[:, prompt_len - 1, :] = (
            modified[:, prompt_len - 1, :] + deltas.to(hidden.device, hidden.dtype)
        )
        applied['done'] = True
        return replace_hidden(output, modified)

    return edit_hook


def append_target(inputs: dict[str, torch.Tensor], target_ids: list[int]) -> dict[str, torch.Tensor]:
    prompt_ids = inputs['input_ids']
    target = torch.tensor(target_ids, dtype=prompt_ids.dtype, device=prompt_ids.device).unsqueeze(0)
    attention = inputs.get('attention_mask', torch.ones_like(prompt_ids))
    target_mask = torch.ones((1, len(target_ids)), dtype=attention.dtype, device=attention.device)
    return {
        'input_ids': torch.cat([prompt_ids, target], dim=1),
        'attention_mask': torch.cat([attention, target_mask], dim=1),
    }


def make_random_controls(delta: torch.Tensor, seed: int, count: int) -> list[torch.Tensor]:
    if count < 1:
        raise ValueError('--random-controls must be at least 1.')
    return [
        normalized_random_control(delta, seed=int(seed) + 104729 * idx)
        for idx in range(int(count))
    ]


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    sizes = sorted({int(size) for size in args.sizes if int(size) > 0})
    if not sizes:
        raise ValueError('At least one positive feature-set size is required.')

    tasks = load_jsonl(args.tasks)
    selected = load_feature_sets(args.catalog, max(sizes))
    missing = sorted({task['concept'] for task in tasks}.difference(selected))
    if missing:
        raise RuntimeError(f'No feature sets for concepts: {missing}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dtype = torch.float16 if device.type == 'cuda' else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(SETTINGS.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        SETTINGS.model_id,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    layers = sorted({int(item['layer']) for item in selected.values()})
    sae_store = SAEStore(
        SETTINGS.sae_repo_id,
        layers=layers,
        device=device,
        dtype=torch.float32,
        top_k=SETTINGS.sae_top_k,
    )

    marker = _completion_marker(args.output)
    if args.resume:
        results: list[dict] = _load_checkpoint_rows(args.output)
    else:
        results = []
        args.output.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)

    completed_counts: dict[str, int] = {}
    for row in results:
        task_id = str(row.get('task_id', ''))
        completed_counts[task_id] = completed_counts.get(task_id, 0) + 1

    for task_idx, task in enumerate(tasks):
        concept = task['concept']
        layer = int(selected[concept]['layer'])
        candidate_ids = [int(x) for x in selected[concept]['feature_ids']]
        valid_sizes = [size for size in sizes if size <= len(candidate_ids)]
        expected_rows = len(valid_sizes) * (1 + int(args.random_controls))
        task_id = str(task['id'])
        if args.resume and completed_counts.get(task_id, 0) == expected_rows:
            print(f"SKIP feature-set task {task_idx + 1}/{len(tasks)}: {task_id}", flush=True)
            continue
        if args.resume and completed_counts.get(task_id, 0):
            results = [row for row in results if str(row.get('task_id', '')) != task_id]
        sae = sae_store.get(layer)
        prompt_inputs = tokenizer(task['prompt'], return_tensors='pt', truncation=True, max_length=192)
        prompt_inputs = {key: value.to(device) for key, value in prompt_inputs.items()}
        prompt_len = int(prompt_inputs['input_ids'].shape[1])
        target_ids = tokenizer(task['target'], add_special_tokens=False)['input_ids']
        if not target_ids:
            raise RuntimeError(f"Target tokenization empty for task {task['id']}")
        target_ids = [int(x) for x in target_ids]
        full_inputs = append_target(prompt_inputs, target_ids)
        capture: dict = {}
        handle = model.model.layers[layer].register_forward_hook(
            _make_capture_hook(capture)
        )
        single_baseline_out = model(**full_inputs, use_cache=False)
        handle.remove()
        single_logits = single_baseline_out.logits[0]
        _, single_mean, _ = sequence_logprob_summary(
            single_logits,
            prompt_length=prompt_len,
            target_ids=target_ids,
        )
        residual = capture['hidden'][0, prompt_len - 1]
        encoding = sae.encode(residual)

        condition_meta: list[tuple[int, str, int, list[int], torch.Tensor]] = []
        for size in valid_sizes:
            feature_ids = candidate_ids[:size]
            activations = [encoding.activation_for(feature_id) for feature_id in feature_ids]
            directions = torch.stack([sae.decoder_direction(feature_id) for feature_id in feature_ids])
            delta, _ = joint_residual_delta(
                directions,
                activations,
                InterventionSpec('ablate', 0.0),
            )
            condition_meta.append((size, 'sae_feature_set', -1, feature_ids, delta))
            controls = make_random_controls(
                delta,
                seed=args.seed + task_idx * 1009 + size * 100_003,
                count=args.random_controls,
            )
            for control_id, control in enumerate(controls):
                condition_meta.append(
                    (size, 'random_norm_matched', control_id, feature_ids, control)
                )

        zero = torch.zeros_like(condition_meta[0][4])
        deltas = torch.stack([zero, *[item[4] for item in condition_meta]], dim=0)
        repeated = {key: value.repeat(deltas.shape[0], 1) for key, value in full_inputs.items()}
        applied = {'done': False}
        hook = model.model.layers[layer].register_forward_hook(
            _make_batch_edit_hook(applied, prompt_len, deltas)
        )
        edited_out = model(**repeated, use_cache=False)
        hook.remove()

        baseline_logits = edited_out.logits[0]
        baseline_next = baseline_logits[prompt_len - 1]
        baseline_seq, baseline_mean, _ = sequence_logprob_summary(
            baseline_logits,
            prompt_length=prompt_len,
            target_ids=target_ids,
        )
        execution_drift_mean = float(baseline_mean - single_mean)
        execution_drift_js = js_divergence_from_logits(single_logits[prompt_len - 1], baseline_next)

        for row_idx, (size, condition, control_id, feature_ids, applied_delta) in enumerate(
            condition_meta,
            start=1,
        ):
            logits = edited_out.logits[row_idx]
            seq_logp, mean_logp, _ = sequence_logprob_summary(
                logits,
                prompt_length=prompt_len,
                target_ids=target_ids,
            )
            active_count = sum(encoding.activation_for(feature_id) > 0 for feature_id in feature_ids)
            results.append(
                {
                    'task_id': task['id'],
                    'concept': concept,
                    'prompt': task['prompt'],
                    'target_text': task['target'],
                    'target_token_count': len(target_ids),
                    'layer': layer,
                    'set_size': int(size),
                    'feature_ids': ','.join(str(x) for x in feature_ids),
                    'active_selected_features': int(active_count),
                    'condition': condition,
                    'control_id': control_id,
                    'random_control_count': args.random_controls,
                    'perturbation_l2': float(torch.linalg.vector_norm(applied_delta.float()).item()),
                    'execution_context_mean_logprob_drift': execution_drift_mean,
                    'execution_context_js_drift': execution_drift_js,
                    'baseline_target_sequence_logprob': baseline_seq,
                    'modified_target_sequence_logprob': seq_logp,
                    'target_sequence_logprob_delta': seq_logp - baseline_seq,
                    'baseline_target_mean_logprob': baseline_mean,
                    'modified_target_mean_logprob': mean_logp,
                    'target_mean_logprob_delta': mean_logp - baseline_mean,
                    'js_divergence': js_divergence_from_logits(
                        baseline_next,
                        logits[prompt_len - 1],
                    ),
                }
            )
        _write_rows_atomic(args.output, results)
        print(f"Feature-set task {task_idx + 1}/{len(tasks)}: {concept}", flush=True)

    _write_rows_atomic(args.output, results)
    marker.write_text('complete\n', encoding='utf-8')
    print(f'Wrote {len(results)} feature-set rows to {args.output}')


if __name__ == '__main__':
    main()