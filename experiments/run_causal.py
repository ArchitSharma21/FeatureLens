from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.common import ARTIFACT_DIR, DATA_DIR, load_jsonl, set_seed
from featurelens.config import SETTINGS
from featurelens.interventions import InterventionSpec, normalized_random_control, residual_delta
from featurelens.metrics import js_divergence_from_logits, sequence_logprob_summary
from featurelens.sae import SAEStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run held-out causal SAE interventions.')
    parser.add_argument('--tasks', type=Path, default=DATA_DIR / 'causal_tasks.jsonl')
    parser.add_argument('--catalog', type=Path, default=ARTIFACT_DIR / 'feature_catalog.csv')
    parser.add_argument('--output', type=Path, default=ARTIFACT_DIR / 'causal_results.csv')
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


def load_selected_features(path: Path) -> dict[str, dict]:
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    selected: dict[str, dict] = {}
    for row in rows:
        concept = row['concept']
        score = float(row['train_auroc'])
        contrast = float(row['activation_rate_pos']) - float(row['activation_rate_neg'])
        key = (score, contrast)
        if concept not in selected or key > selected[concept]['_key']:
            selected[concept] = {
                '_key': key,
                'layer': int(row['layer']),
                'feature_id': int(row['feature_id']),
                'train_auroc': score,
                'test_auroc': float(row['auroc']),
                'test_f1': float(row['f1']),
            }
    return selected


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

    def batch_edit_hook(_module, _inp, output):
        if applied['done']:
            return output
        hidden = hidden_from_output(output)
        modified = hidden.clone()
        modified[:, prompt_len - 1, :] = (
            modified[:, prompt_len - 1, :] + deltas.to(hidden.device, hidden.dtype)
        )
        applied['done'] = True
        return replace_hidden(output, modified)

    return batch_edit_hook


def append_target(inputs: dict[str, torch.Tensor], target_ids: list[int]) -> dict[str, torch.Tensor]:
    prompt_ids = inputs['input_ids']
    target = torch.tensor(target_ids, dtype=prompt_ids.dtype, device=prompt_ids.device).unsqueeze(0)
    full_ids = torch.cat([prompt_ids, target], dim=1)
    attention = inputs.get('attention_mask', torch.ones_like(prompt_ids))
    target_mask = torch.ones((1, len(target_ids)), dtype=attention.dtype, device=attention.device)
    return {
        'input_ids': full_ids,
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
    tasks = load_jsonl(args.tasks)
    selected = load_selected_features(args.catalog)
    missing = sorted({task['concept'] for task in tasks}.difference(selected))
    if missing:
        raise RuntimeError(f'No selected SAE features for concepts: {missing}')

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
    selected_layers = sorted({item['layer'] for item in selected.values()})
    sae_store = SAEStore(
        SETTINGS.sae_repo_id,
        layers=selected_layers,
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

    expected_rows_per_task = 2 * (1 + int(args.random_controls))
    completed_counts: dict[str, int] = {}
    for row in results:
        task_id = str(row.get('task_id', ''))
        completed_counts[task_id] = completed_counts.get(task_id, 0) + 1

    for task_idx, task in enumerate(tasks):
        task_id = str(task['id'])
        if args.resume and completed_counts.get(task_id, 0) == expected_rows_per_task:
            print(f"SKIP causal task {task_idx + 1}/{len(tasks)}: {task_id}", flush=True)
            continue
        if args.resume and completed_counts.get(task_id, 0):
            results = [row for row in results if str(row.get('task_id', '')) != task_id]

        concept = task['concept']
        choice = selected[concept]
        layer = int(choice['layer'])
        feature_id = int(choice['feature_id'])
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
        single_baseline_logits = single_baseline_out.logits[0]
        _, single_baseline_mean, _ = sequence_logprob_summary(
            single_baseline_logits,
            prompt_length=prompt_len,
            target_ids=target_ids,
        )

        residual = capture['hidden'][0, prompt_len - 1]
        encoding = sae.encode(residual)
        original_activation = encoding.activation_for(feature_id)

        specs = [
            ('ablate', InterventionSpec('ablate', 0.0)),
            ('amplify_2x', InterventionSpec('scale', 2.0)),
        ]
        condition_meta: list[tuple[str, str, int, InterventionSpec, torch.Tensor, float]] = []
        for spec_idx, (intervention_name, spec) in enumerate(specs):
            delta = residual_delta(sae.decoder_direction(feature_id), original_activation, spec)
            condition_meta.append(
                (
                    intervention_name,
                    'sae_feature',
                    -1,
                    spec,
                    delta,
                    float(spec.delta_activation(original_activation)),
                )
            )
            controls = make_random_controls(
                delta,
                seed=args.seed + task_idx * 1009 + spec_idx * 100_003,
                count=args.random_controls,
            )
            for control_id, control_delta in enumerate(controls):
                condition_meta.append(
                    (
                        intervention_name,
                        'random_norm_matched',
                        control_id,
                        spec,
                        control_delta,
                        math.nan,
                    )
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
        execution_drift_mean = float(baseline_mean - single_baseline_mean)
        execution_drift_js = js_divergence_from_logits(
            single_baseline_logits[prompt_len - 1], baseline_next
        )
        target_id = target_ids[0]
        baseline_prob = float(torch.softmax(baseline_next.float(), dim=-1)[target_id].item())
        baseline_rank = int((baseline_next > baseline_next[target_id]).sum().item()) + 1
        baseline_top1 = int(torch.argmax(baseline_next).item())

        for row_idx, (intervention_name, condition, control_id, _spec, applied_delta, delta_activation) in enumerate(
            condition_meta,
            start=1,
        ):
            modified_logits = edited_out.logits[row_idx]
            modified_next = modified_logits[prompt_len - 1]
            modified_prob = float(torch.softmax(modified_next.float(), dim=-1)[target_id].item())
            modified_rank = int((modified_next > modified_next[target_id]).sum().item()) + 1
            modified_top1 = int(torch.argmax(modified_next).item())
            modified_seq, modified_mean, _ = sequence_logprob_summary(
                modified_logits,
                prompt_length=prompt_len,
                target_ids=target_ids,
            )
            results.append(
                {
                    'task_id': task['id'],
                    'concept': concept,
                    'prompt': task['prompt'],
                    'target_text': task['target'],
                    'target_first_token': tokenizer.decode([target_id]),
                    'target_token_count': len(target_ids),
                    'layer': layer,
                    'feature_id': feature_id,
                    'feature_train_auroc': choice['train_auroc'],
                    'feature_test_auroc': choice['test_auroc'],
                    'feature_test_f1': choice['test_f1'],
                    'feature_activation': original_activation,
                    'intervention': intervention_name,
                    'condition': condition,
                    'control_id': control_id,
                    'random_control_count': args.random_controls,
                    'delta_activation': delta_activation,
                    'perturbation_l2': float(torch.linalg.vector_norm(applied_delta.float()).item()),
                    'execution_context_mean_logprob_drift': execution_drift_mean,
                    'execution_context_js_drift': execution_drift_js,
                    'baseline_target_prob': baseline_prob,
                    'modified_target_prob': modified_prob,
                    'target_prob_delta': modified_prob - baseline_prob,
                    'target_logprob_delta': float(
                        torch.log_softmax(modified_next.float(), dim=-1)[target_id].item()
                        - torch.log_softmax(baseline_next.float(), dim=-1)[target_id].item()
                    ),
                    'baseline_target_rank': baseline_rank,
                    'modified_target_rank': modified_rank,
                    'target_rank_delta': modified_rank - baseline_rank,
                    'baseline_target_sequence_logprob': baseline_seq,
                    'modified_target_sequence_logprob': modified_seq,
                    'target_sequence_logprob_delta': modified_seq - baseline_seq,
                    'baseline_target_mean_logprob': baseline_mean,
                    'modified_target_mean_logprob': modified_mean,
                    'target_mean_logprob_delta': modified_mean - baseline_mean,
                    'js_divergence': js_divergence_from_logits(baseline_next, modified_next),
                    'top1_changed': int(modified_top1 != baseline_top1),
                }
            )
        _write_rows_atomic(args.output, results)
        print(f"Causal task {task_idx + 1}/{len(tasks)}: {concept}", flush=True)

    _write_rows_atomic(args.output, results)
    marker.write_text('complete\n', encoding='utf-8')
    print(f'Wrote {len(results)} causal intervention rows to {args.output}')


if __name__ == '__main__':
    main()
