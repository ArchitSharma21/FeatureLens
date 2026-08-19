from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download


@dataclass
class SparseEncoding:
    indices: torch.Tensor
    values: torch.Tensor
    pre_activations: torch.Tensor | None = None

    @property
    def active_count(self) -> int:
        return int((self.values > 0).sum().item())

    def activation_for(self, feature_id: int) -> float:
        mask = self.indices == int(feature_id)
        if not bool(mask.any()):
            return 0.0
        return float(self.values[mask][0].item())


@dataclass
class SAEWeights:
    layer: int
    w_enc_t: torch.Tensor  # [d_model, d_sae]
    w_dec: torch.Tensor  # [d_model, d_sae]
    b_enc: torch.Tensor  # [d_sae]
    b_dec: torch.Tensor  # [d_model]
    top_k: int = 50

    @torch.inference_mode()
    def encode(self, hidden: torch.Tensor, return_pre: bool = False) -> SparseEncoding:
        """Encode one or more residual vectors without materializing dense sparse acts."""
        if hidden.shape[-1] != self.w_enc_t.shape[0]:
            raise ValueError(
                f'Expected hidden dim {self.w_enc_t.shape[0]}, got {hidden.shape[-1]}.'
            )
        compute_hidden = hidden.to(device=self.w_enc_t.device, dtype=self.w_enc_t.dtype)
        pre = compute_hidden @ self.w_enc_t + self.b_enc
        relu = torch.relu(pre)
        values, indices = torch.topk(relu, k=self.top_k, dim=-1)
        return SparseEncoding(indices=indices, values=values, pre_activations=pre if return_pre else None)

    @torch.inference_mode()
    def decode_sparse(self, encoding: SparseEncoding) -> torch.Tensor:
        """Decode TopK features efficiently using only selected decoder columns."""
        indices = encoding.indices
        values = encoding.values.to(device=self.w_dec.device, dtype=self.w_dec.dtype)
        if indices.ndim == 1:
            cols = self.w_dec[:, indices]  # [d_model, k]
            return self.b_dec + cols @ values
        flat_idx = indices.reshape(-1, indices.shape[-1])
        flat_vals = values.reshape(-1, values.shape[-1])
        outputs = []
        for row_idx, row_vals in zip(flat_idx, flat_vals, strict=True):
            cols = self.w_dec[:, row_idx]
            outputs.append(self.b_dec + cols @ row_vals)
        return torch.stack(outputs).reshape(*indices.shape[:-1], self.w_dec.shape[0])

    def decoder_direction(self, feature_id: int) -> torch.Tensor:
        if feature_id < 0 or feature_id >= self.w_dec.shape[1]:
            raise ValueError(f'Feature id must be in [0, {self.w_dec.shape[1] - 1}].')
        return self.w_dec[:, int(feature_id)]


class SAEStore:
    def __init__(
        self,
        repo_id: str,
        layers: Iterable[int],
        device: torch.device,
        dtype: torch.dtype,
        top_k: int = 50,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.layers = tuple(int(x) for x in layers)
        self.device = device
        self.dtype = dtype
        self.top_k = int(top_k)
        self.cache_dir = str(cache_dir) if cache_dir else None
        self._cache: dict[int, SAEWeights] = {}

    def get(self, layer: int) -> SAEWeights:
        layer = int(layer)
        if layer not in self.layers:
            raise ValueError(f'Layer {layer} is not configured. Available: {self.layers}.')
        if layer in self._cache:
            return self._cache[layer]

        path = hf_hub_download(
            repo_id=self.repo_id,
            filename=f'layer{layer}.sae.pt',
            cache_dir=self.cache_dir,
        )
        try:
            raw = torch.load(path, map_location='cpu', weights_only=True)
        except TypeError:  # pragma: no cover - old torch fallback
            raw = torch.load(path, map_location='cpu')

        required = {'W_enc', 'W_dec', 'b_enc', 'b_dec'}
        missing = required.difference(raw)
        if missing:
            raise KeyError(f'SAE checkpoint layer {layer} missing keys: {sorted(missing)}')

        sae = SAEWeights(
            layer=layer,
            w_enc_t=raw['W_enc'].T.contiguous().to(self.device, dtype=self.dtype),
            w_dec=raw['W_dec'].contiguous().to(self.device, dtype=self.dtype),
            b_enc=raw['b_enc'].contiguous().to(self.device, dtype=self.dtype),
            b_dec=raw['b_dec'].contiguous().to(self.device, dtype=self.dtype),
            top_k=self.top_k,
        )
        self._cache[layer] = sae
        return sae

    def preload(self) -> None:
        for layer in self.layers:
            self.get(layer)
