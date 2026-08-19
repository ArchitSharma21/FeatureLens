from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.common import ARTIFACT_DIR, DATA_DIR, load_jsonl, set_seed
from featurelens.config import SETTINGS
from featurelens.metrics import reconstruction_metrics
from featurelens.sae import SAEStore, SparseEncoding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Collect residual and Qwen-Scope SAE activations.')
    parser.add_argument('--input', type=Path, default=DATA_DIR / 'prompts.jsonl')
    parser.add_argument('--output-dir', type=Path, default=ARTIFACT_DIR / 'activations')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-length', type=int, default=192)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--layers', type=int, nargs='+', default=list(SETTINGS.layers))
    return parser.parse_args()


def _build_sparse(encodings: list[SparseEncoding], n_rows: int, width: int) -> sp.csr_matrix:
    row_ids: list[int] = []
    col_ids: list[int] = []
    values: list[float] = []
    for row, encoding in enumerate(encodings):
        idx = encoding.indices.detach().cpu().numpy().reshape(-1)
        vals = encoding.values.detach().float().cpu().numpy().reshape(-1)
        positive = vals > 0
        row_ids.extend([row] * int(positive.sum()))
        col_ids.extend(idx[positive].astype(int).tolist())
        values.extend(vals[positive].astype(float).tolist())
    return sp.csr_matrix((values, (row_ids, col_ids)), shape=(n_rows, width), dtype=np.float32)


def _promptwide_max_encoding(
    token_encoding: SparseEncoding,
    attention_mask: torch.Tensor,
) -> list[SparseEncoding]:
    """Max-pool sparse feature activations across non-padding tokens for each prompt."""
    indices = token_encoding.indices.detach().cpu()
    values = token_encoding.values.detach().float().cpu()
    mask = attention_mask.detach().bool().cpu()
    pooled: list[SparseEncoding] = []

    for row_idx in range(indices.shape[0]):
        feature_max: dict[int, float] = {}
        valid_positions = torch.nonzero(mask[row_idx], as_tuple=False).reshape(-1).tolist()
        for token_idx in valid_positions:
            token_ids = indices[row_idx, token_idx].reshape(-1).tolist()
            token_values = values[row_idx, token_idx].reshape(-1).tolist()
            for feature_id, activation in zip(token_ids, token_values, strict=True):
                activation = float(activation)
                if activation <= 0.0:
                    continue
                feature_id = int(feature_id)
                if activation > feature_max.get(feature_id, 0.0):
                    feature_max[feature_id] = activation

        if feature_max:
            ordered = sorted(feature_max.items())
            pooled.append(
                SparseEncoding(
                    indices=torch.tensor([item[0] for item in ordered], dtype=torch.long),
                    values=torch.tensor([item[1] for item in ordered], dtype=torch.float32),
                )
            )
        else:
            pooled.append(
                SparseEncoding(
                    indices=torch.empty(0, dtype=torch.long),
                    values=torch.empty(0, dtype=torch.float32),
                )
            )
    return pooled


def _make_capture_hook(
    captured: dict[int, torch.Tensor],
    layer: int,
):
    """Bind the capture dictionary and layer before registering the hook."""

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured[layer] = hidden.detach()

    return hook


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rows = load_jsonl(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dtype = torch.float16 if device.type == 'cuda' else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(SETTINGS.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        SETTINGS.model_id,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    sae_store = SAEStore(
        SETTINGS.sae_repo_id,
        layers=args.layers,
        device=device,
        dtype=torch.float32,
        top_k=SETTINGS.sae_top_k,
    )

    residuals: dict[int, list[np.ndarray]] = {layer: [] for layer in args.layers}
    final_encodings: dict[int, list[SparseEncoding]] = {layer: [] for layer in args.layers}
    promptwide_encodings: dict[int, list[SparseEncoding]] = {layer: [] for layer in args.layers}
    recon_stats: dict[int, list[dict[str, float]]] = {layer: [] for layer in args.layers}

    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        texts = [row['text'] for row in batch_rows]
        batch = tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        captured: dict[int, torch.Tensor] = {}
        handles = []

        for layer in args.layers:
            handles.append(
                model.model.layers[layer].register_forward_hook(
                    _make_capture_hook(captured, layer)
                )
            )
        model(**batch, use_cache=False)
        for handle in handles:
            handle.remove()

        for layer in args.layers:
            sae = sae_store.get(layer)
            hidden = captured[layer]
            final_token_residuals = hidden[:, -1, :]
            final_batch_encoding = sae.encode(final_token_residuals)
            token_batch_encoding = sae.encode(hidden)
            promptwide_batch = _promptwide_max_encoding(
                token_batch_encoding,
                batch['attention_mask'],
            )

            for row_idx in range(final_token_residuals.shape[0]):
                residual = final_token_residuals[row_idx]
                final_encoding = SparseEncoding(
                    indices=final_batch_encoding.indices[row_idx],
                    values=final_batch_encoding.values[row_idx],
                )
                reconstructed = sae.decode_sparse(final_encoding)
                residuals[layer].append(
                    residual.detach().float().cpu().numpy().astype(np.float16)
                )
                final_encodings[layer].append(final_encoding)
                promptwide_encodings[layer].append(promptwide_batch[row_idx])
                recon_stats[layer].append(reconstruction_metrics(residual, reconstructed))

        print(f'Processed {min(start + args.batch_size, len(rows))}/{len(rows)} prompts', flush=True)

    for layer in args.layers:
        residual_array = np.stack(residuals[layer], axis=0)
        np.save(args.output_dir / f'residuals_layer{layer}.npy', residual_array)

        promptwide_sparse = _build_sparse(
            promptwide_encodings[layer],
            len(rows),
            SETTINGS.sae_width,
        )
        sp.save_npz(
            args.output_dir / f'features_layer{layer}.npz',
            promptwide_sparse,
            compressed=True,
        )

        final_sparse = _build_sparse(
            final_encodings[layer],
            len(rows),
            SETTINGS.sae_width,
        )
        sp.save_npz(
            args.output_dir / f'features_final_layer{layer}.npz',
            final_sparse,
            compressed=True,
        )

        summary = {
            'layer': layer,
            'n_samples': len(rows),
            'mean_cosine': float(np.mean([item['cosine'] for item in recon_stats[layer]])),
            'mean_nmse': float(np.mean([item['nmse'] for item in recon_stats[layer]])),
            'median_nmse': float(np.median([item['nmse'] for item in recon_stats[layer]])),
            'mean_active_features_final_token': float(np.mean(np.diff(final_sparse.indptr))),
            'mean_active_features_promptwide': float(np.mean(np.diff(promptwide_sparse.indptr))),
        }
        # Preserve the legacy key for report compatibility. It refers to final-token TopK activity.
        summary['mean_active_features'] = summary['mean_active_features_final_token']
        (args.output_dir / f'reconstruction_layer{layer}.json').write_text(
            json.dumps(summary, indent=2),
            encoding='utf-8',
        )

    metadata = {
        'model_id': SETTINGS.model_id,
        'sae_repo_id': SETTINGS.sae_repo_id,
        'layers': args.layers,
        'top_k': SETTINGS.sae_top_k,
        'width': SETTINGS.sae_width,
        'n_samples': len(rows),
        'feature_pooling': 'prompt-wide max activation across non-padding tokens',
        'feature_file_pattern': 'features_layer{layer}.npz',
        'final_token_feature_file_pattern': 'features_final_layer{layer}.npz',
        'dense_residual_pooling': 'final prompt token',
        'rows': rows,
    }
    (args.output_dir / 'metadata.json').write_text(
        json.dumps(metadata, indent=2),
        encoding='utf-8',
    )
    print(f'Activation artifacts written to {args.output_dir}')


if __name__ == '__main__':
    main()
