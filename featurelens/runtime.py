from __future__ import annotations

import hashlib
import html
import json
import math
import os
import random
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .catalog import FeatureCatalog
from .config import SETTINGS, Settings
from .interventions import (
    InterventionSpec,
    joint_residual_delta,
    normalized_random_control,
    residual_delta,
)
from .metrics import (
    contrastive_log_odds,
    decoder_cosine_matrix,
    joint_direction_norm_ratio,
    js_divergence_from_logits,
    reconstruction_metrics,
    sequence_logprob_summary,
    sparse_topk_cosine,
)
from .sae import SAEStore, SparseEncoding


def _dtype_from_name(name: str) -> torch.dtype:
    name = name.lower().strip()
    if name in {'float16', 'fp16', 'half'}:
        return torch.float16
    if name in {'bfloat16', 'bf16'}:
        return torch.bfloat16
    if name in {'float32', 'fp32'}:
        return torch.float32
    raise ValueError(f'Unsupported dtype: {name}')


def _default_device() -> torch.device:
    # ZeroGPU exposes CUDA emulation at module load time. Hugging Face recommends
    # placing models on CUDA at module scope so startup transfers can be optimized.
    if os.getenv('SPACE_ID'):
        return torch.device('cuda')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


@dataclass
class AnalysisResult:
    tokens: list[str]
    token_index: int
    layer: int
    features: SparseEncoding
    rows: list[list[object]]
    metrics: dict[str, float]


@dataclass
class InterventionResult:
    baseline_text: str
    modified_text: str
    feature_activation: float
    delta_activation: float
    perturbation_norm: float
    js_divergence: float
    random_js_divergence: float
    random_js_std: float
    js_specificity_ratio: float
    js_empirical_p: float
    random_control_count: int
    execution_drift_js: float
    execution_drift_mean_logprob: float | None
    target_text: str
    target_token_count: int
    target_tokens: list[str]
    baseline_target_prob: float | None
    modified_target_prob: float | None
    random_target_prob: float | None
    baseline_sequence_logprob: float | None
    modified_sequence_logprob: float | None
    random_sequence_logprob: float | None
    sequence_logprob_delta: float | None
    random_sequence_logprob_delta: float | None
    mean_logprob_delta: float | None
    random_mean_logprob_delta: float | None
    random_abs_mean_logprob_delta: float | None
    random_mean_logprob_std: float | None
    target_specificity_ratio: float | None
    target_empirical_p: float | None
    target_token_rows: list[list[object]]
    top_token_rows: list[list[object]]


@dataclass
class LayerSweepResult:
    tokens: list[str]
    token_index: int
    rows: list[list[object]]


@dataclass
class DoseResponseResult:
    feature_activation: float
    target_tokens: list[str]
    execution_drift_mean_logprob: float
    execution_drift_js: float
    rows: list[list[object]]


@dataclass
class FeatureSetResult:
    feature_ids: list[int]
    feature_rows: list[list[object]]
    perturbation_norm: float
    js_divergence: float
    random_js_divergence: float
    random_js_std: float
    js_specificity_ratio: float
    js_empirical_p: float
    random_control_count: int
    execution_drift_mean_logprob: float
    execution_drift_js: float
    baseline_sequence_logprob: float
    modified_sequence_logprob: float
    random_sequence_logprob: float
    sequence_logprob_delta: float
    random_sequence_logprob_delta: float
    mean_logprob_delta: float
    random_mean_logprob_delta: float
    random_abs_mean_logprob_delta: float
    random_mean_logprob_std: float
    target_specificity_ratio: float
    target_empirical_p: float
    target_tokens: list[str]
    target_token_rows: list[list[object]]


@dataclass
class FeatureSetSweepResult:
    target_tokens: list[str]
    random_control_count: int
    execution_drift_mean_logprob: float
    execution_drift_js: float
    rows: list[list[object]]


@dataclass
class FeatureInteractionResult:
    feature_ids: list[int]
    target_tokens: list[str]
    rows: list[list[object]]
    additive_expected_mean_delta: float
    joint_mean_delta: float
    interaction_excess_mean_delta: float
    normalized_interaction: float
    execution_drift_mean_logprob: float


@dataclass
class ConceptContrastResult:
    feature_id: int
    layer: int
    prompts_per_concept: int
    rows: list[list[object]]
    chart_rows: list[list[object]]
    leading_concept: str | None
    leading_ratio: float | None
    active_prompt_count: int
    total_prompt_count: int


@dataclass
class FeatureTraceResult:
    feature_id: int
    layer: int
    tokens: list[str]
    rows: list[list[object]]
    chart_rows: list[list[object]]
    active_token_count: int
    token_count: int
    max_activation: float
    max_token_index: int | None


@dataclass
class FeatureGeometryResult:
    feature_ids: list[int]
    layer: int
    rows: list[list[object]]
    chart_rows: list[list[object]]
    mean_abs_decoder_cosine: float
    max_abs_decoder_cosine: float
    joint_ablation_norm: float
    independent_norm: float
    alignment_ratio: float


@dataclass
class ContrastiveCausalResult:
    feature_id: int
    layer: int
    feature_activation: float
    perturbation_norm: float
    target_a_tokens: list[str]
    target_b_tokens: list[str]
    rows: list[list[object]]
    baseline_log_odds: float
    modified_log_odds: float
    delta_log_odds: float
    baseline_normalized_preference: float
    modified_normalized_preference: float
    delta_normalized_preference: float
    random_signed_mean_delta: float
    random_abs_mean_delta: float
    random_delta_std: float
    specificity_ratio: float
    empirical_p: float
    random_control_count: int


@dataclass
class ConceptFeatureDiscoveryResult:
    concept: str
    layer: int
    prompts_per_concept: int
    top_n: int
    ranking_mode: str
    rows: list[list[object]]
    chart_rows: list[list[object]]
    candidate_ids: list[int]
    default_candidate_id: int | None
    current_context_available: bool
    current_token_index: int | None
    displayed_current_active_count: int
    split_half_k: int | None
    split_half_shared_count: int
    split_half_jaccard: float | None
    resample_replicates: int
    resample_mean_support: float | None
    resample_high_support_count: int


@dataclass
class CandidateCausalScreenResult:
    feature_ids: list[int]
    target_tokens: list[str]
    rows: list[list[object]]
    chart_rows: list[list[object]]
    active_feature_count: int
    candidate_count: int
    execution_drift_mean_logprob: float
    execution_drift_js: float


@dataclass
class CandidateSpecificityResult:
    feature_ids: list[int]
    target_tokens: list[str]
    rows: list[list[object]]
    chart_rows: list[list[object]]
    active_feature_count: int
    candidate_count: int
    random_control_count: int
    execution_drift_mean_logprob: float
    execution_drift_js: float


@dataclass
class CandidateCrossTargetResult:
    feature_ids: list[int]
    targets: list[str]
    rows: list[list[object]]
    chart_rows: list[list[object]]
    summary_rows: list[list[object]]
    pairwise_rows: list[list[object]]
    active_feature_count: int


@dataclass
class FeatureCueScanResult:
    feature_id: int
    layer: int
    prompt_stem: str
    rows: list[list[object]]
    chart_rows: list[list[object]]
    active_cue_count: int
    cue_count: int


@dataclass
class FeatureCueContextResult:
    feature_id: int
    layer: int
    stems: list[str]
    cues: list[str]
    rows: list[list[object]]
    chart_rows: list[list[object]]
    active_condition_count: int
    condition_count: int
    cue_active_context_counts: dict[str, int]
    cue_mean_activations: dict[str, float]
    dominant_cue: str | None
    dominant_cue_context_count: int
    off_dominant_active_count: int


@dataclass
class ParaphraseResult:
    tokens_a: list[str]
    token_index_a: int
    tokens_b: list[str]
    token_index_b: int
    topk_jaccard: float
    sparse_cosine: float
    promptwide_jaccard: float
    promptwide_cosine: float
    shared_top_n: int
    top_n: int
    rows: list[list[object]]
    chart_rows: list[list[object]]


class FeatureLensRuntime:
    def __init__(self, settings: Settings = SETTINGS) -> None:
        self.settings = settings
        self.device = _default_device()
        self.model_dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        self.sae_dtype = _dtype_from_name(settings.sae_dtype)
        if self.device.type == 'cpu' and self.sae_dtype != torch.float32:
            self.sae_dtype = torch.float32
        self.model = None
        self.tokenizer = None
        self.sae_store: SAEStore | None = None
        self.catalog = FeatureCatalog()
        self.load_error: str | None = None

    @property
    def ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None and self.sae_store is not None

    def ensure_ready(self, preload_saes: bool = False) -> None:
        if self.ready:
            assert self.sae_store is not None
            if preload_saes:
                self.sae_store.preload()
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.settings.model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.model = AutoModelForCausalLM.from_pretrained(
            self.settings.model_id,
            torch_dtype=self.model_dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.sae_store = SAEStore(
            repo_id=self.settings.sae_repo_id,
            layers=self.settings.layers,
            device=self.device,
            dtype=self.sae_dtype,
            top_k=self.settings.sae_top_k,
        )
        if preload_saes:
            self.sae_store.preload()
        self.load_error = None

    def token_choices(self, text: str) -> list[tuple[str, int]]:
        self.ensure_ready(preload_saes=False)
        assert self.tokenizer is not None
        ids = self.tokenizer(text, add_special_tokens=True)['input_ids']
        tokens = [self.tokenizer.decode([token_id]) for token_id in ids]
        return [(f'{idx}: {token!r}', idx) for idx, token in enumerate(tokens)]

    def _inputs(self, text: str) -> dict[str, torch.Tensor]:
        assert self.tokenizer is not None
        batch = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=self.settings.max_prompt_tokens,
        )
        return {key: value.to(self.device) for key, value in batch.items()}

    def _target_ids(self, target_text: str) -> list[int]:
        assert self.tokenizer is not None
        ids = self.tokenizer(target_text, add_special_tokens=False)['input_ids']
        if not ids:
            raise ValueError('Target continuation tokenized to an empty sequence.')
        return [int(x) for x in ids]

    def _append_target(
        self,
        prompt_inputs: dict[str, torch.Tensor],
        target_ids: Sequence[int],
    ) -> dict[str, torch.Tensor]:
        prompt_ids = prompt_inputs['input_ids']
        target = torch.tensor(
            list(target_ids),
            dtype=prompt_ids.dtype,
            device=prompt_ids.device,
        ).unsqueeze(0)
        full_ids = torch.cat([prompt_ids, target], dim=1)
        if 'attention_mask' in prompt_inputs:
            target_mask = torch.ones(
                (prompt_ids.shape[0], len(target_ids)),
                dtype=prompt_inputs['attention_mask'].dtype,
                device=prompt_ids.device,
            )
            attention = torch.cat([prompt_inputs['attention_mask'], target_mask], dim=1)
        else:
            attention = torch.ones_like(full_ids)
        return {'input_ids': full_ids, 'attention_mask': attention}

    @staticmethod
    def _repeat_inputs(inputs: dict[str, torch.Tensor], repeats: int) -> dict[str, torch.Tensor]:
        return {key: value.repeat(int(repeats), 1) for key, value in inputs.items()}

    @staticmethod
    def _hidden_from_output(output):
        return output[0] if isinstance(output, tuple) else output

    @staticmethod
    def _replace_hidden_in_output(output, hidden: torch.Tensor):
        if isinstance(output, tuple):
            return (hidden, *output[1:])
        return hidden

    @contextmanager
    def _capture_hook(self, layer: int, bucket: dict) -> Iterator[None]:
        assert self.model is not None

        def hook(_module, _inputs, output):
            hidden = self._hidden_from_output(output)
            if 'hidden' not in bucket:
                bucket['hidden'] = hidden.detach()

        handle = self.model.model.layers[int(layer)].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @contextmanager
    def _capture_hooks(self, layers: Sequence[int], buckets: dict[int, dict]) -> Iterator[None]:
        assert self.model is not None
        handles = []
        for layer in layers:
            bucket = buckets[int(layer)]

            def hook(_module, _inputs, output, *, target=bucket):
                hidden = self._hidden_from_output(output)
                if 'hidden' not in target:
                    target['hidden'] = hidden.detach()

            handles.append(self.model.model.layers[int(layer)].register_forward_hook(hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def _delta_hook(self, layer: int, token_index: int, delta: torch.Tensor) -> Iterator[None]:
        assert self.model is not None
        applied = {'done': False}

        def hook(_module, _inputs, output):
            if applied['done']:
                return output
            hidden = self._hidden_from_output(output)
            if hidden.ndim != 3:
                return output
            seq_len = hidden.shape[1]
            idx = self._resolve_index(int(token_index), seq_len)
            modified = hidden.clone()
            modified[:, idx, :] = modified[:, idx, :] + delta.to(hidden.device, hidden.dtype)
            applied['done'] = True
            return self._replace_hidden_in_output(output, modified)

        handle = self.model.model.layers[int(layer)].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @contextmanager
    def _batch_delta_hook(
        self,
        layer: int,
        token_index: int,
        deltas: torch.Tensor,
    ) -> Iterator[None]:
        """Apply one residual delta per batch row in a single model forward."""
        assert self.model is not None
        if deltas.ndim != 2:
            raise ValueError('deltas must have shape [batch, d_model].')
        applied = {'done': False}

        def hook(_module, _inputs, output):
            if applied['done']:
                return output
            hidden = self._hidden_from_output(output)
            if hidden.ndim != 3:
                return output
            if hidden.shape[0] != deltas.shape[0]:
                raise ValueError('Delta batch size does not match model batch size.')
            idx = self._resolve_index(int(token_index), hidden.shape[1])
            modified = hidden.clone()
            modified[:, idx, :] = modified[:, idx, :] + deltas.to(hidden.device, hidden.dtype)
            applied['done'] = True
            return self._replace_hidden_in_output(output, modified)

        handle = self.model.model.layers[int(layer)].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @staticmethod
    def _resolve_index(token_index: int, seq_len: int) -> int:
        idx = int(token_index)
        if idx < 0:
            idx = seq_len + idx
        if idx < 0 or idx >= seq_len:
            raise IndexError(f'Token index {token_index} outside prompt length {seq_len}.')
        return idx

    @staticmethod
    def _control_seed(text: str, layer: int, key: str, mode: str, coefficient: float) -> int:
        payload = f'{text}\0{layer}\0{key}\0{mode}\0{coefficient:.8g}'.encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'big', signed=False)

    @staticmethod
    def _random_controls(delta: torch.Tensor, seed: int, count: int) -> list[torch.Tensor]:
        if int(count) < 1:
            raise ValueError('Random-control ensemble must contain at least one direction.')
        return [
            normalized_random_control(delta, seed=int(seed) + 104729 * idx)
            for idx in range(int(count))
        ]

    @staticmethod
    def _random_effect_summary(values: Sequence[float], target_effect: float) -> tuple[float, float, float, float]:
        if not values:
            raise ValueError('Random-control values must not be empty.')
        tensor = torch.tensor([float(x) for x in values], dtype=torch.float64)
        signed_mean = float(tensor.mean().item())
        abs_mean = float(tensor.abs().mean().item())
        std = float(tensor.std(unbiased=False).item())
        empirical_p = float(
            (1 + int((tensor.abs() >= abs(float(target_effect))).sum().item()))
            / (len(values) + 1)
        )
        return signed_mean, abs_mean, std, empirical_p

    @staticmethod
    def _dict_cosine(a: dict[int, float], b: dict[int, float]) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        dot = sum(float(value) * float(b.get(feature_id, 0.0)) for feature_id, value in a.items())
        norm_a = math.sqrt(sum(float(value) ** 2 for value in a.values()))
        norm_b = math.sqrt(sum(float(value) ** 2 for value in b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    @staticmethod
    def _dict_jaccard(a: dict[int, float], b: dict[int, float]) -> float:
        set_a = {feature_id for feature_id, value in a.items() if float(value) > 0}
        set_b = {feature_id for feature_id, value in b.items() if float(value) > 0}
        union = set_a | set_b
        return float(len(set_a & set_b) / len(union)) if union else 1.0

    @staticmethod
    def _max_pool_encoding(encoding: SparseEncoding) -> dict[int, float]:
        indices = encoding.indices.detach().cpu()
        values = encoding.values.detach().float().cpu()
        if indices.ndim == 1:
            indices = indices.unsqueeze(0)
            values = values.unsqueeze(0)
        pooled: dict[int, float] = {}
        for row_ids, row_values in zip(indices.tolist(), values.tolist(), strict=True):
            for feature_id, value in zip(row_ids, row_values, strict=True):
                value = float(value)
                if value <= 0:
                    continue
                feature_id = int(feature_id)
                pooled[feature_id] = max(pooled.get(feature_id, 0.0), value)
        return pooled

    @staticmethod
    def _encoding_map(encoding: SparseEncoding) -> dict[int, float]:
        return {
            int(feature_id): float(value)
            for feature_id, value in zip(
                encoding.indices.detach().cpu().tolist(),
                encoding.values.detach().float().cpu().tolist(),
                strict=True,
            )
            if float(value) > 0
        }

    @torch.inference_mode()
    def _analyze_and_pool(
        self,
        text: str,
        layer: int,
        token_index: int = -1,
        top_n: int = 12,
    ) -> tuple[AnalysisResult, dict[int, float]]:
        self.ensure_ready(preload_saes=False)
        assert self.model is not None and self.tokenizer is not None and self.sae_store is not None
        if int(layer) not in self.settings.layers:
            raise ValueError(f'Layer must be one of {self.settings.layers}.')
        inputs = self._inputs(text)
        bucket: dict = {}
        with self._capture_hook(int(layer), bucket):
            self.model(**inputs, use_cache=False)
        hidden = bucket['hidden'][0]
        idx = self._resolve_index(int(token_index), hidden.shape[0])
        sae = self.sae_store.get(int(layer))
        all_encoding = sae.encode(hidden)
        encoding = SparseEncoding(
            indices=all_encoding.indices[idx],
            values=all_encoding.values[idx],
        )
        residual = hidden[idx]
        reconstruction = sae.decode_sparse(encoding)
        metrics = reconstruction_metrics(residual, reconstruction)
        metrics['active_features'] = float(encoding.active_count)
        values = encoding.values.float().clamp_min(0)
        total = float(values.sum().item())
        metrics['top5_mass_fraction'] = (
            float(values[: min(5, values.numel())].sum().item()) / total if total > 0 else 0.0
        )
        ids = inputs['input_ids'][0].tolist()
        tokens = [self.tokenizer.decode([token_id]) for token_id in ids]
        rows: list[list[object]] = []
        rank_count = min(int(top_n), encoding.indices.numel())
        for rank in range(rank_count):
            feature_id = int(encoding.indices[rank].item())
            activation = float(encoding.values[rank].item())
            rows.append(
                [rank + 1, feature_id, activation, self.catalog.hint(int(layer), feature_id)]
            )
        result = AnalysisResult(
            tokens=tokens,
            token_index=idx,
            layer=int(layer),
            features=encoding,
            rows=rows,
            metrics=metrics,
        )
        return result, self._max_pool_encoding(all_encoding)

    @torch.inference_mode()
    def analyze(self, text: str, layer: int, token_index: int = -1, top_n: int = 12) -> AnalysisResult:
        result, _ = self._analyze_and_pool(text, layer, token_index, top_n)
        return result

    @torch.inference_mode()
    def layer_sweep(self, text: str, token_index: int = -1) -> LayerSweepResult:
        self.ensure_ready(preload_saes=True)
        assert self.model is not None and self.tokenizer is not None and self.sae_store is not None
        inputs = self._inputs(text)
        buckets = {int(layer): {} for layer in self.settings.layers}
        with self._capture_hooks(self.settings.layers, buckets):
            self.model(**inputs, use_cache=False)

        seq_len = int(inputs['input_ids'].shape[1])
        idx = self._resolve_index(int(token_index), seq_len)
        rows: list[list[object]] = []
        for layer in self.settings.layers:
            residual = buckets[int(layer)]['hidden'][0, idx]
            sae = self.sae_store.get(int(layer))
            encoding = sae.encode(residual)
            reconstruction = sae.decode_sparse(encoding)
            metrics = reconstruction_metrics(residual, reconstruction)
            values = encoding.values.float().clamp_min(0)
            positive = values[values > 0]
            total = positive.sum()
            if positive.numel() <= 1 or float(total.item()) <= 0:
                entropy = 0.0
            else:
                probs = positive / total
                entropy = float(
                    (-(probs * torch.log(probs)).sum() / math.log(positive.numel())).item()
                )
            top5_fraction = (
                float(values[: min(5, values.numel())].sum().item() / total.item())
                if float(total.item()) > 0
                else 0.0
            )
            rows.append(
                [
                    int(layer),
                    float(metrics['cosine']),
                    float(metrics['nmse']),
                    int(encoding.active_count),
                    float(values[0].item()) if values.numel() else 0.0,
                    top5_fraction,
                    entropy,
                ]
            )
        ids = inputs['input_ids'][0].tolist()
        tokens = [self.tokenizer.decode([token_id]) for token_id in ids]
        return LayerSweepResult(tokens=tokens, token_index=idx, rows=rows)

    def token_html(self, tokens: list[str], selected_index: int) -> str:
        chips = []
        for idx, token in enumerate(tokens):
            safe = html.escape(token if token.strip() else repr(token))
            selected = idx == int(selected_index)
            cls = 'token selected' if selected else 'token'
            chips.append(f'<span class="{cls}"><sup>{idx}</sup>{safe}</span>')
        return '<div class="token-wrap">' + ''.join(chips) + '</div>'

    @staticmethod
    def _top_token_rows(tokenizer, baseline_logits: torch.Tensor, modified_logits: torch.Tensor, k: int = 8):
        p = torch.softmax(baseline_logits.float(), dim=-1)
        q = torch.softmax(modified_logits.float(), dim=-1)
        union_ids = torch.unique(torch.cat([torch.topk(p, k).indices, torch.topk(q, k).indices]))
        rows = []
        for token_id in union_ids.tolist():
            token = tokenizer.decode([int(token_id)])
            bp = float(p[token_id].item())
            mp = float(q[token_id].item())
            rows.append([repr(token), bp, mp, mp - bp])
        rows.sort(key=lambda row: max(row[1], row[2]), reverse=True)
        return rows[: min(len(rows), 12)]

    def _target_rows(
        self,
        target_ids: Sequence[int],
        baseline_token_logps: Sequence[float],
        modified_token_logps: Sequence[float],
        random_token_logps: Sequence[float],
    ) -> list[list[object]]:
        assert self.tokenizer is not None
        rows = []
        for idx, (token_id, bp, mp, rp) in enumerate(
            zip(
                target_ids,
                baseline_token_logps,
                modified_token_logps,
                random_token_logps,
                strict=True,
            )
        ):
            rows.append(
                [
                    idx,
                    repr(self.tokenizer.decode([int(token_id)])),
                    float(bp),
                    float(mp),
                    float(rp),
                    float(mp - bp),
                    float(rp - bp),
                ]
            )
        return rows

    @torch.inference_mode()
    def intervene(
        self,
        text: str,
        layer: int,
        token_index: int,
        feature_id: int,
        mode: str,
        coefficient: float,
        target_text: str = '',
        max_new_tokens: int = 24,
    ) -> InterventionResult:
        self.ensure_ready(preload_saes=False)
        assert self.model is not None and self.tokenizer is not None and self.sae_store is not None
        prompt_inputs = self._inputs(text)
        prompt_len = int(prompt_inputs['input_ids'].shape[1])
        idx = self._resolve_index(int(token_index), prompt_len)
        sae = self.sae_store.get(int(layer))

        target_ids: list[int] = []
        capture: dict = {}
        single_baseline_logits: torch.Tensor
        single_baseline_mean: float | None = None

        if target_text.strip():
            target_ids = self._target_ids(target_text)
            scoring_inputs = self._append_target(prompt_inputs, target_ids)
            with self._capture_hook(int(layer), capture):
                single_baseline_out = self.model(**scoring_inputs, use_cache=False)
            single_baseline_logits = single_baseline_out.logits[0]
            _, single_baseline_mean, _ = sequence_logprob_summary(
                single_baseline_logits,
                prompt_length=prompt_len,
                target_ids=target_ids,
            )
        else:
            scoring_inputs = prompt_inputs
            with self._capture_hook(int(layer), capture):
                single_baseline_out = self.model(**scoring_inputs, use_cache=False)
            single_baseline_logits = single_baseline_out.logits[0]

        residual = capture['hidden'][0, idx]
        encoding = sae.encode(residual)
        original_activation = encoding.activation_for(int(feature_id))
        spec = InterventionSpec(mode=mode, coefficient=float(coefficient))
        delta = residual_delta(sae.decoder_direction(int(feature_id)), original_activation, spec)
        seed = self._control_seed(
            text, int(layer), str(int(feature_id)), mode, float(coefficient)
        )
        random_controls = self._random_controls(
            delta, seed=seed, count=self.settings.live_random_controls
        )

        zero = torch.zeros_like(delta)
        all_deltas = torch.stack([zero, delta, *random_controls], dim=0)
        repeated = self._repeat_inputs(scoring_inputs, all_deltas.shape[0])
        with self._batch_delta_hook(int(layer), idx, all_deltas):
            edited = self.model(**repeated, use_cache=False)

        baseline_logits = edited.logits[0]
        modified_logits = edited.logits[1]
        random_logits = [edited.logits[row] for row in range(2, edited.logits.shape[0])]
        next_idx = prompt_len - 1
        baseline_next_logits = baseline_logits[next_idx]
        modified_next_logits = modified_logits[next_idx]
        random_next_logits = [row[next_idx] for row in random_logits]

        execution_drift_js = js_divergence_from_logits(
            single_baseline_logits[next_idx], baseline_next_logits
        )
        js = js_divergence_from_logits(baseline_next_logits, modified_next_logits)
        random_js_values = [
            js_divergence_from_logits(baseline_next_logits, logits)
            for logits in random_next_logits
        ]
        random_js_mean, random_js_abs_mean, random_js_std, js_empirical_p = (
            self._random_effect_summary(random_js_values, js)
        )
        # JS divergence is non-negative, so signed and absolute means are identical up to numerical noise.
        random_js_reference = random_js_abs_mean
        js_ratio = abs(js) / max(random_js_reference, 1e-12)

        baseline_seq = baseline_mean = modified_seq = modified_mean = None
        random_seq_mean = random_mean_signed = random_abs_mean = random_mean_std = None
        sequence_delta = random_sequence_delta = mean_delta = specificity = target_p = None
        target_rows: list[list[object]] = []
        target_tokens: list[str] = []
        bp = mp = rp = None
        execution_drift_mean = None

        if target_ids:
            baseline_seq, baseline_mean, baseline_token_logps = sequence_logprob_summary(
                baseline_logits, prompt_length=prompt_len, target_ids=target_ids
            )
            modified_seq, modified_mean, modified_token_logps = sequence_logprob_summary(
                modified_logits, prompt_length=prompt_len, target_ids=target_ids
            )
            random_summaries = [
                sequence_logprob_summary(logits, prompt_length=prompt_len, target_ids=target_ids)
                for logits in random_logits
            ]
            random_seqs = [item[0] for item in random_summaries]
            random_means = [item[1] for item in random_summaries]
            random_token_matrix = [item[2] for item in random_summaries]
            random_token_mean = [
                float(sum(row[token_pos] for row in random_token_matrix) / len(random_token_matrix))
                for token_pos in range(len(target_ids))
            ]
            random_mean_deltas = [float(value - baseline_mean) for value in random_means]
            random_seq_deltas = [float(value - baseline_seq) for value in random_seqs]
            mean_delta = float(modified_mean - baseline_mean)
            sequence_delta = float(modified_seq - baseline_seq)
            random_mean_signed, random_abs_mean, random_mean_std, target_p = self._random_effect_summary(
                random_mean_deltas, mean_delta
            )
            random_sequence_delta = float(sum(random_seq_deltas) / len(random_seq_deltas))
            random_seq_mean = float(sum(random_seqs) / len(random_seqs))
            specificity = abs(mean_delta) / max(random_abs_mean, 1e-12)
            _, single_mean, _ = sequence_logprob_summary(
                single_baseline_logits, prompt_length=prompt_len, target_ids=target_ids
            )
            execution_drift_mean = float(baseline_mean - single_mean)

            p = torch.softmax(baseline_next_logits.float(), dim=-1)
            q = torch.softmax(modified_next_logits.float(), dim=-1)
            random_probs = [torch.softmax(logits.float(), dim=-1) for logits in random_next_logits]
            first_id = int(target_ids[0])
            bp = float(p[first_id].item())
            mp = float(q[first_id].item())
            rp = float(sum(prob[first_id].item() for prob in random_probs) / len(random_probs))
            target_tokens = [self.tokenizer.decode([int(token_id)]) for token_id in target_ids]
            target_rows = self._target_rows(
                target_ids, baseline_token_logps, modified_token_logps, random_token_mean
            )

        generation_kwargs = {
            'max_new_tokens': min(int(max_new_tokens), self.settings.max_new_tokens),
            'do_sample': False,
            'return_dict_in_generate': True,
            'output_scores': False,
            'pad_token_id': self.tokenizer.eos_token_id,
        }
        baseline_generation = self.model.generate(**prompt_inputs, **generation_kwargs)
        with self._delta_hook(int(layer), idx, delta):
            modified_generation = self.model.generate(**prompt_inputs, **generation_kwargs)
        baseline_ids = baseline_generation.sequences[0, prompt_len:]
        modified_ids = modified_generation.sequences[0, prompt_len:]
        baseline_text = self.tokenizer.decode(baseline_ids, skip_special_tokens=True)
        modified_text = self.tokenizer.decode(modified_ids, skip_special_tokens=True)

        return InterventionResult(
            baseline_text=baseline_text,
            modified_text=modified_text,
            feature_activation=float(original_activation),
            delta_activation=float(spec.delta_activation(original_activation)),
            perturbation_norm=float(torch.linalg.vector_norm(delta.float()).item()),
            js_divergence=float(js),
            random_js_divergence=float(random_js_reference),
            random_js_std=float(random_js_std),
            js_specificity_ratio=float(js_ratio),
            js_empirical_p=float(js_empirical_p),
            random_control_count=len(random_controls),
            execution_drift_js=float(execution_drift_js),
            execution_drift_mean_logprob=execution_drift_mean,
            target_text=target_text,
            target_token_count=len(target_ids),
            target_tokens=target_tokens,
            baseline_target_prob=bp,
            modified_target_prob=mp,
            random_target_prob=rp,
            baseline_sequence_logprob=baseline_seq,
            modified_sequence_logprob=modified_seq,
            random_sequence_logprob=random_seq_mean,
            sequence_logprob_delta=sequence_delta,
            random_sequence_logprob_delta=random_sequence_delta,
            mean_logprob_delta=mean_delta,
            random_mean_logprob_delta=random_mean_signed,
            random_abs_mean_logprob_delta=random_abs_mean,
            random_mean_logprob_std=random_mean_std,
            target_specificity_ratio=specificity,
            target_empirical_p=target_p,
            target_token_rows=target_rows,
            top_token_rows=self._top_token_rows(
                self.tokenizer, baseline_next_logits, modified_next_logits, k=8
            ),
        )

    @torch.inference_mode()
    def dose_response(
        self,
        text: str,
        layer: int,
        token_index: int,
        feature_id: int,
        target_text: str,
        multipliers: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
    ) -> DoseResponseResult:
        if not target_text.strip():
            raise ValueError('Dose-response requires a target continuation.')
        if not any(abs(float(multiplier) - 1.0) < 1e-12 for multiplier in multipliers):
            raise ValueError('Dose-response multipliers must include 1.0 as the zero-edit reference.')
        self.ensure_ready(preload_saes=False)
        assert self.model is not None and self.tokenizer is not None and self.sae_store is not None
        prompt_inputs = self._inputs(text)
        prompt_len = int(prompt_inputs['input_ids'].shape[1])
        idx = self._resolve_index(int(token_index), prompt_len)
        target_ids = self._target_ids(target_text)
        full_inputs = self._append_target(prompt_inputs, target_ids)
        sae = self.sae_store.get(int(layer))

        capture: dict = {}
        with self._capture_hook(int(layer), capture):
            single_baseline_out = self.model(**full_inputs, use_cache=False)
        single_baseline_logits = single_baseline_out.logits[0]
        _, single_baseline_mean, _ = sequence_logprob_summary(
            single_baseline_logits, prompt_length=prompt_len, target_ids=target_ids
        )
        residual = capture['hidden'][0, idx]
        encoding = sae.encode(residual)
        original_activation = encoding.activation_for(int(feature_id))
        direction = sae.decoder_direction(int(feature_id))

        deltas: list[torch.Tensor] = []
        delta_coefficients: list[float] = []
        norms: list[float] = []
        for multiplier in multipliers:
            spec = InterventionSpec('scale', float(multiplier))
            delta = residual_delta(direction, original_activation, spec)
            deltas.append(delta)
            delta_coefficients.append(float(spec.delta_activation(original_activation)))
            norms.append(float(torch.linalg.vector_norm(delta.float()).item()))

        repeated = self._repeat_inputs(full_inputs, len(deltas))
        with self._batch_delta_hook(int(layer), idx, torch.stack(deltas, dim=0)):
            outputs = self.model(**repeated, use_cache=False)

        reference_idx = next(
            idx for idx, multiplier in enumerate(multipliers) if abs(float(multiplier) - 1.0) < 1e-12
        )
        reference_logits = outputs.logits[reference_idx]
        baseline_seq, baseline_mean, _ = sequence_logprob_summary(
            reference_logits, prompt_length=prompt_len, target_ids=target_ids
        )
        baseline_next = reference_logits[prompt_len - 1]
        execution_drift_mean = float(baseline_mean - single_baseline_mean)
        execution_drift_js = js_divergence_from_logits(
            single_baseline_logits[prompt_len - 1], baseline_next
        )

        rows: list[list[object]] = []
        for row_idx, multiplier in enumerate(multipliers):
            modified_logits = outputs.logits[row_idx]
            modified_seq, modified_mean, _ = sequence_logprob_summary(
                modified_logits, prompt_length=prompt_len, target_ids=target_ids
            )
            rows.append(
                [
                    float(multiplier),
                    delta_coefficients[row_idx],
                    norms[row_idx],
                    float(baseline_mean),
                    float(modified_mean),
                    float(modified_mean - baseline_mean),
                    float(modified_seq - baseline_seq),
                    float(js_divergence_from_logits(baseline_next, modified_logits[prompt_len - 1])),
                ]
            )
        return DoseResponseResult(
            feature_activation=float(original_activation),
            target_tokens=[self.tokenizer.decode([int(token_id)]) for token_id in target_ids],
            execution_drift_mean_logprob=execution_drift_mean,
            execution_drift_js=float(execution_drift_js),
            rows=rows,
        )

    @torch.inference_mode()
    def intervene_feature_set(
        self,
        text: str,
        layer: int,
        token_index: int,
        feature_ids: Sequence[int],
        mode: str,
        coefficient: float,
        target_text: str,
    ) -> FeatureSetResult:
        if not target_text.strip():
            raise ValueError('Feature-set causal testing requires a target continuation.')
        ids = list(dict.fromkeys(int(x) for x in feature_ids))
        if not ids:
            raise ValueError('Select at least one feature.')
        if len(ids) > 12:
            raise ValueError('Select at most 12 features for a live feature-set intervention.')
        if mode not in {'ablate', 'scale'}:
            raise ValueError("Feature-set mode must be 'ablate' or 'scale'.")

        self.ensure_ready(preload_saes=False)
        assert self.model is not None and self.tokenizer is not None and self.sae_store is not None
        prompt_inputs = self._inputs(text)
        prompt_len = int(prompt_inputs['input_ids'].shape[1])
        idx = self._resolve_index(int(token_index), prompt_len)
        target_ids = self._target_ids(target_text)
        full_inputs = self._append_target(prompt_inputs, target_ids)
        sae = self.sae_store.get(int(layer))

        capture: dict = {}
        with self._capture_hook(int(layer), capture):
            single_baseline_out = self.model(**full_inputs, use_cache=False)
        single_baseline_logits = single_baseline_out.logits[0]
        _, single_baseline_mean, _ = sequence_logprob_summary(
            single_baseline_logits, prompt_length=prompt_len, target_ids=target_ids
        )
        residual = capture['hidden'][0, idx]
        encoding = sae.encode(residual)
        activations = [encoding.activation_for(feature_id) for feature_id in ids]
        directions = torch.stack([sae.decoder_direction(feature_id) for feature_id in ids], dim=0)
        spec = InterventionSpec(mode, float(coefficient))
        delta, coefficient_deltas = joint_residual_delta(directions, activations, spec)
        seed = self._control_seed(
            text, int(layer), ','.join(str(x) for x in ids), mode, float(coefficient)
        )
        controls = self._random_controls(
            delta, seed=seed, count=self.settings.live_random_controls
        )

        zero = torch.zeros_like(delta)
        all_deltas = torch.stack([zero, delta, *controls], dim=0)
        repeated = self._repeat_inputs(full_inputs, all_deltas.shape[0])
        with self._batch_delta_hook(int(layer), idx, all_deltas):
            outputs = self.model(**repeated, use_cache=False)

        baseline_logits = outputs.logits[0]
        modified_logits = outputs.logits[1]
        random_logits = [outputs.logits[row] for row in range(2, outputs.logits.shape[0])]
        baseline_seq, baseline_mean, baseline_tokens = sequence_logprob_summary(
            baseline_logits, prompt_length=prompt_len, target_ids=target_ids
        )
        modified_seq, modified_mean, modified_tokens = sequence_logprob_summary(
            modified_logits, prompt_length=prompt_len, target_ids=target_ids
        )
        random_summaries = [
            sequence_logprob_summary(logits, prompt_length=prompt_len, target_ids=target_ids)
            for logits in random_logits
        ]
        random_seqs = [item[0] for item in random_summaries]
        random_means = [item[1] for item in random_summaries]
        random_token_matrix = [item[2] for item in random_summaries]
        random_token_mean = [
            float(sum(row[token_pos] for row in random_token_matrix) / len(random_token_matrix))
            for token_pos in range(len(target_ids))
        ]
        mean_delta = float(modified_mean - baseline_mean)
        sequence_delta = float(modified_seq - baseline_seq)
        random_mean_deltas = [float(value - baseline_mean) for value in random_means]
        random_seq_deltas = [float(value - baseline_seq) for value in random_seqs]
        random_mean_signed, random_abs_mean, random_mean_std, target_p = self._random_effect_summary(
            random_mean_deltas, mean_delta
        )
        next_idx = prompt_len - 1
        js = js_divergence_from_logits(baseline_logits[next_idx], modified_logits[next_idx])
        random_js_values = [
            js_divergence_from_logits(baseline_logits[next_idx], logits[next_idx])
            for logits in random_logits
        ]
        _, random_js_abs_mean, random_js_std, js_p = self._random_effect_summary(
            random_js_values, js
        )
        execution_drift_mean = float(baseline_mean - single_baseline_mean)
        execution_drift_js = js_divergence_from_logits(
            single_baseline_logits[next_idx], baseline_logits[next_idx]
        )

        feature_rows = [
            [
                feature_id,
                float(activation),
                float(delta_coefficient),
                self.catalog.hint(int(layer), feature_id),
            ]
            for feature_id, activation, delta_coefficient in zip(
                ids, activations, coefficient_deltas, strict=True
            )
        ]
        return FeatureSetResult(
            feature_ids=ids,
            feature_rows=feature_rows,
            perturbation_norm=float(torch.linalg.vector_norm(delta.float()).item()),
            js_divergence=float(js),
            random_js_divergence=float(random_js_abs_mean),
            random_js_std=float(random_js_std),
            js_specificity_ratio=float(abs(js) / max(random_js_abs_mean, 1e-12)),
            js_empirical_p=float(js_p),
            random_control_count=len(controls),
            execution_drift_mean_logprob=execution_drift_mean,
            execution_drift_js=float(execution_drift_js),
            baseline_sequence_logprob=float(baseline_seq),
            modified_sequence_logprob=float(modified_seq),
            random_sequence_logprob=float(sum(random_seqs) / len(random_seqs)),
            sequence_logprob_delta=sequence_delta,
            random_sequence_logprob_delta=float(sum(random_seq_deltas) / len(random_seq_deltas)),
            mean_logprob_delta=mean_delta,
            random_mean_logprob_delta=random_mean_signed,
            random_abs_mean_logprob_delta=random_abs_mean,
            random_mean_logprob_std=float(random_mean_std),
            target_specificity_ratio=float(abs(mean_delta) / max(random_abs_mean, 1e-12)),
            target_empirical_p=float(target_p),
            target_tokens=[self.tokenizer.decode([int(token_id)]) for token_id in target_ids],
            target_token_rows=self._target_rows(
                target_ids, baseline_tokens, modified_tokens, random_token_mean
            ),
        )

    @torch.inference_mode()
    def feature_set_size_sweep(
        self,
        text: str,
        layer: int,
        token_index: int,
        target_text: str,
        sizes: Sequence[int] = (1, 3, 5),
    ) -> FeatureSetSweepResult:
        """Jointly ablate the strongest k active features and compare to random ensembles."""
        if not target_text.strip():
            raise ValueError('Feature-set size sweep requires a target continuation.')
        self.ensure_ready(preload_saes=False)
        assert self.model is not None and self.tokenizer is not None and self.sae_store is not None
        prompt_inputs = self._inputs(text)
        prompt_len = int(prompt_inputs['input_ids'].shape[1])
        idx = self._resolve_index(int(token_index), prompt_len)
        target_ids = self._target_ids(target_text)
        full_inputs = self._append_target(prompt_inputs, target_ids)
        sae = self.sae_store.get(int(layer))

        capture: dict = {}
        with self._capture_hook(int(layer), capture):
            single_baseline_out = self.model(**full_inputs, use_cache=False)
        single_baseline_logits = single_baseline_out.logits[0]
        _, single_baseline_mean, _ = sequence_logprob_summary(
            single_baseline_logits, prompt_length=prompt_len, target_ids=target_ids
        )
        residual = capture['hidden'][0, idx]
        encoding = sae.encode(residual)
        active_ids = [
            int(feature_id)
            for feature_id, value in zip(
                encoding.indices.detach().cpu().tolist(),
                encoding.values.detach().float().cpu().tolist(),
                strict=True,
            )
            if float(value) > 0
        ]
        valid_sizes = [int(size) for size in sizes if int(size) > 0 and int(size) <= len(active_ids)]
        if not valid_sizes:
            raise ValueError('Not enough active features for the requested set sizes.')

        condition_deltas: list[torch.Tensor] = [torch.zeros_like(residual)]
        metadata: list[tuple[int, str, list[int], float]] = []
        for size in valid_sizes:
            selected = active_ids[:size]
            activations = [encoding.activation_for(feature_id) for feature_id in selected]
            directions = torch.stack([sae.decoder_direction(feature_id) for feature_id in selected])
            delta, _ = joint_residual_delta(
                directions, activations, InterventionSpec('ablate', 0.0)
            )
            norm = float(torch.linalg.vector_norm(delta.float()).item())
            condition_deltas.append(delta)
            metadata.append((size, 'sae', selected, norm))
            seed = self._control_seed(text, int(layer), f'top-{size}', 'ablate_set', 0.0)
            controls = self._random_controls(
                delta, seed=seed, count=self.settings.live_random_controls
            )
            for control_idx, control in enumerate(controls):
                condition_deltas.append(control)
                metadata.append((size, f'random_{control_idx}', selected, norm))

        all_deltas = torch.stack(condition_deltas, dim=0)
        repeated = self._repeat_inputs(full_inputs, all_deltas.shape[0])
        with self._batch_delta_hook(int(layer), idx, all_deltas):
            outputs = self.model(**repeated, use_cache=False)

        baseline_logits = outputs.logits[0]
        baseline_seq, baseline_mean, _ = sequence_logprob_summary(
            baseline_logits, prompt_length=prompt_len, target_ids=target_ids
        )
        baseline_next = baseline_logits[prompt_len - 1]
        execution_drift_mean = float(baseline_mean - single_baseline_mean)
        execution_drift_js = js_divergence_from_logits(
            single_baseline_logits[prompt_len - 1], baseline_next
        )

        grouped: dict[int, dict[str, object]] = {
            size: {'features': active_ids[:size], 'norm': None, 'sae': None, 'random': []}
            for size in valid_sizes
        }
        for output_idx, meta in enumerate(metadata, start=1):
            size, kind, selected, norm = meta
            logits = outputs.logits[output_idx]
            seq_logp, mean_logp, _ = sequence_logprob_summary(
                logits, prompt_length=prompt_len, target_ids=target_ids
            )
            item = {
                'seq': float(seq_logp),
                'mean': float(mean_logp),
                'js': float(js_divergence_from_logits(baseline_next, logits[prompt_len - 1])),
            }
            grouped[size]['norm'] = norm
            grouped[size]['features'] = selected
            if kind == 'sae':
                grouped[size]['sae'] = item
            else:
                grouped[size]['random'].append(item)

        rows: list[list[object]] = []
        for size in valid_sizes:
            group = grouped[size]
            sae_item = group['sae']
            random_items = group['random']
            assert isinstance(sae_item, dict)
            assert isinstance(random_items, list) and random_items
            sae_mean_delta = float(sae_item['mean'] - baseline_mean)
            sae_seq_delta = float(sae_item['seq'] - baseline_seq)
            random_mean_deltas = [float(item['mean'] - baseline_mean) for item in random_items]
            random_js_values = [float(item['js']) for item in random_items]
            random_signed, random_abs, random_std, target_p = self._random_effect_summary(
                random_mean_deltas, sae_mean_delta
            )
            _, random_js_abs, random_js_std, js_p = self._random_effect_summary(
                random_js_values, float(sae_item['js'])
            )
            rows.append(
                [
                    int(size),
                    ', '.join(str(x) for x in group['features']),
                    float(group['norm']),
                    float(baseline_mean),
                    float(sae_item['mean']),
                    sae_mean_delta,
                    random_signed,
                    random_abs,
                    random_std,
                    float(abs(sae_mean_delta) / max(random_abs, 1e-12)),
                    float(target_p),
                    sae_seq_delta,
                    float(sae_item['js']),
                    random_js_abs,
                    random_js_std,
                    float(js_p),
                ]
            )
        return FeatureSetSweepResult(
            target_tokens=[self.tokenizer.decode([int(token_id)]) for token_id in target_ids],
            random_control_count=self.settings.live_random_controls,
            execution_drift_mean_logprob=execution_drift_mean,
            execution_drift_js=float(execution_drift_js),
            rows=rows,
        )

    @torch.inference_mode()
    def feature_interaction_test(
        self,
        text: str,
        layer: int,
        token_index: int,
        feature_ids: Sequence[int],
        target_text: str,
    ) -> FeatureInteractionResult:
        """Compare individual ablations with their joint ablation to measure non-additivity."""
        if not target_text.strip():
            raise ValueError('Feature interaction testing requires a target continuation.')
        ids = list(dict.fromkeys(int(x) for x in feature_ids))
        if len(ids) < 2:
            raise ValueError('Select at least two features for the interaction decomposition.')
        if len(ids) > 5:
            raise ValueError('Select at most five features for the live interaction decomposition.')
        self.ensure_ready(preload_saes=False)
        assert self.model is not None and self.tokenizer is not None and self.sae_store is not None
        prompt_inputs = self._inputs(text)
        prompt_len = int(prompt_inputs['input_ids'].shape[1])
        idx = self._resolve_index(int(token_index), prompt_len)
        target_ids = self._target_ids(target_text)
        full_inputs = self._append_target(prompt_inputs, target_ids)
        sae = self.sae_store.get(int(layer))

        capture: dict = {}
        with self._capture_hook(int(layer), capture):
            single_baseline_out = self.model(**full_inputs, use_cache=False)
        single_logits = single_baseline_out.logits[0]
        _, single_mean, _ = sequence_logprob_summary(
            single_logits, prompt_length=prompt_len, target_ids=target_ids
        )
        residual = capture['hidden'][0, idx]
        encoding = sae.encode(residual)
        activations = [encoding.activation_for(feature_id) for feature_id in ids]
        directions = [sae.decoder_direction(feature_id) for feature_id in ids]
        individual_deltas = [
            residual_delta(direction, activation, InterventionSpec('ablate', 0.0))
            for direction, activation in zip(directions, activations, strict=True)
        ]
        joint_delta = torch.stack(individual_deltas, dim=0).sum(dim=0)
        all_deltas = torch.stack([torch.zeros_like(joint_delta), *individual_deltas, joint_delta], dim=0)
        repeated = self._repeat_inputs(full_inputs, all_deltas.shape[0])
        with self._batch_delta_hook(int(layer), idx, all_deltas):
            outputs = self.model(**repeated, use_cache=False)

        baseline_logits = outputs.logits[0]
        baseline_seq, baseline_mean, _ = sequence_logprob_summary(
            baseline_logits, prompt_length=prompt_len, target_ids=target_ids
        )
        baseline_next = baseline_logits[prompt_len - 1]
        rows: list[list[object]] = []
        individual_mean_deltas: list[float] = []
        for feature_idx, feature_id in enumerate(ids, start=1):
            logits = outputs.logits[feature_idx]
            seq_logp, mean_logp, _ = sequence_logprob_summary(
                logits, prompt_length=prompt_len, target_ids=target_ids
            )
            delta_mean = float(mean_logp - baseline_mean)
            individual_mean_deltas.append(delta_mean)
            rows.append(
                [
                    f'Feature {feature_id}',
                    str(feature_id),
                    float(activations[feature_idx - 1]),
                    float(torch.linalg.vector_norm(individual_deltas[feature_idx - 1].float()).item()),
                    delta_mean,
                    float(seq_logp - baseline_seq),
                    float(js_divergence_from_logits(baseline_next, logits[prompt_len - 1])),
                ]
            )
        joint_logits = outputs.logits[len(ids) + 1]
        joint_seq, joint_mean, _ = sequence_logprob_summary(
            joint_logits, prompt_length=prompt_len, target_ids=target_ids
        )
        joint_mean_delta = float(joint_mean - baseline_mean)
        rows.append(
            [
                'Joint ablation',
                ', '.join(str(x) for x in ids),
                float(sum(activations)),
                float(torch.linalg.vector_norm(joint_delta.float()).item()),
                joint_mean_delta,
                float(joint_seq - baseline_seq),
                float(js_divergence_from_logits(baseline_next, joint_logits[prompt_len - 1])),
            ]
        )
        additive_expected = float(sum(individual_mean_deltas))
        interaction_excess = float(joint_mean_delta - additive_expected)
        scale = max(sum(abs(value) for value in individual_mean_deltas), 1e-12)
        return FeatureInteractionResult(
            feature_ids=ids,
            target_tokens=[self.tokenizer.decode([int(token_id)]) for token_id in target_ids],
            rows=rows,
            additive_expected_mean_delta=additive_expected,
            joint_mean_delta=joint_mean_delta,
            interaction_excess_mean_delta=interaction_excess,
            normalized_interaction=float(interaction_excess / scale),
            execution_drift_mean_logprob=float(baseline_mean - single_mean),
        )

    @torch.inference_mode()
    def compare_paraphrases(
        self,
        text_a: str,
        text_b: str,
        layer: int,
        token_index_a: int = -1,
        token_index_b: int = -1,
        top_n: int = 12,
    ) -> ParaphraseResult:
        if not text_a.strip() or not text_b.strip():
            raise ValueError('Enter both the original prompt and a paraphrase.')
        a, pooled_a = self._analyze_and_pool(text_a, int(layer), int(token_index_a), max(int(top_n), 12))
        b, pooled_b = self._analyze_and_pool(text_b, int(layer), int(token_index_b), max(int(top_n), 12))

        map_a = self._encoding_map(a.features)
        map_b = self._encoding_map(b.features)
        set_a = set(map_a)
        set_b = set(map_b)
        union = set_a | set_b
        jaccard = len(set_a & set_b) / len(union) if union else 1.0
        cosine = sparse_topk_cosine(
            a.features.indices, a.features.values, b.features.indices, b.features.values
        )
        promptwide_jaccard = self._dict_jaccard(pooled_a, pooled_b)
        promptwide_cosine = self._dict_cosine(pooled_a, pooled_b)

        top_ids_a = [int(row[1]) for row in a.rows[: int(top_n)]]
        top_ids_b = [int(row[1]) for row in b.rows[: int(top_n)]]
        top_union = list(dict.fromkeys(top_ids_a + top_ids_b))
        shared_top_n = len(set(top_ids_a) & set(top_ids_b))
        rows: list[list[object]] = []
        chart_rows: list[list[object]] = []
        for feature_id in top_union:
            va = float(map_a.get(feature_id, 0.0))
            vb = float(map_b.get(feature_id, 0.0))
            status = 'shared' if va > 0 and vb > 0 else ('original only' if va > 0 else 'paraphrase only')
            rows.append(
                [feature_id, va, vb, status, self.catalog.hint(int(layer), feature_id)]
            )
            chart_rows.append([str(feature_id), 'Original', va])
            chart_rows.append([str(feature_id), 'Paraphrase', vb])

        rows.sort(key=lambda row: max(float(row[1]), float(row[2])), reverse=True)
        return ParaphraseResult(
            tokens_a=a.tokens,
            token_index_a=a.token_index,
            tokens_b=b.tokens,
            token_index_b=b.token_index,
            topk_jaccard=float(jaccard),
            sparse_cosine=float(cosine),
            promptwide_jaccard=float(promptwide_jaccard),
            promptwide_cosine=float(promptwide_cosine),
            shared_top_n=int(shared_top_n),
            top_n=int(top_n),
            rows=rows,
            chart_rows=chart_rows,
        )

    def _contrast_prompt_rows(self, prompts_per_concept: int) -> list[dict]:
        data_path = Path(__file__).resolve().parents[1] / 'data' / 'prompts.jsonl'
        rows = [
            json.loads(line)
            for line in data_path.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        selected: list[dict] = []
        by_concept: dict[str, list[dict]] = {}
        for row in rows:
            # Use one wording per paraphrase pair so the live contrast is not dominated by near-duplicates.
            if int(row.get('variant', 0)) != 0:
                continue
            by_concept.setdefault(str(row['concept']), []).append(row)
        for concept in sorted(by_concept):
            selected.extend(by_concept[concept][: int(prompts_per_concept)])
        return selected

    @torch.inference_mode()
    def feature_token_trace(
        self,
        text: str,
        layer: int,
        feature_id: int,
    ) -> FeatureTraceResult:
        """Trace one SAE feature across every non-padding token in a prompt."""
        if not text.strip():
            raise ValueError('Enter a prompt first.')
        self.ensure_ready(preload_saes=False)
        assert self.model is not None and self.tokenizer is not None and self.sae_store is not None
        if int(layer) not in self.settings.layers:
            raise ValueError(f'Layer must be one of {self.settings.layers}.')
        if int(feature_id) < 0 or int(feature_id) >= self.settings.sae_width:
            raise ValueError(f'Feature id must be in [0, {self.settings.sae_width - 1}].')

        inputs = self._inputs(text)
        capture: dict = {}
        with self._capture_hook(int(layer), capture):
            self.model(**inputs, use_cache=False)
        hidden = capture['hidden'][0]
        sae = self.sae_store.get(int(layer))
        encoding = sae.encode(hidden)
        mask = encoding.indices == int(feature_id)
        activations = torch.where(mask, encoding.values, torch.zeros_like(encoding.values)).sum(dim=-1)
        values = activations.detach().float().cpu().tolist()
        ids = inputs['input_ids'][0].tolist()
        tokens = [self.tokenizer.decode([int(token_id)]) for token_id in ids]

        rows: list[list[object]] = []
        chart_rows: list[list[object]] = []
        for idx, (token, value) in enumerate(zip(tokens, values, strict=True)):
            value = float(value)
            rows.append([idx, repr(token), value, bool(value > 0)])
            chart_rows.append([f'{idx}: {token if token.strip() else repr(token)}', value])

        active = [idx for idx, value in enumerate(values) if float(value) > 0]
        if active:
            max_idx = max(active, key=lambda idx: float(values[idx]))
            max_value = float(values[max_idx])
        else:
            max_idx = None
            max_value = 0.0
        return FeatureTraceResult(
            feature_id=int(feature_id),
            layer=int(layer),
            tokens=tokens,
            rows=rows,
            chart_rows=chart_rows,
            active_token_count=len(active),
            token_count=len(tokens),
            max_activation=max_value,
            max_token_index=max_idx,
        )

    @torch.inference_mode()
    def feature_geometry(
        self,
        text: str,
        layer: int,
        token_index: int,
        feature_ids: Sequence[int],
    ) -> FeatureGeometryResult:
        """Inspect pairwise SAE decoder geometry and activation-weighted ablation geometry."""
        if not text.strip():
            raise ValueError('Enter a prompt first.')
        ids = list(dict.fromkeys(int(x) for x in feature_ids))
        if len(ids) < 2:
            raise ValueError('Select at least two distinct features for geometry analysis.')
        if len(ids) > 8:
            raise ValueError('Geometry analysis supports at most eight features in the live app.')
        if any(feature_id < 0 or feature_id >= self.settings.sae_width for feature_id in ids):
            raise ValueError(f'Feature ids must be in [0, {self.settings.sae_width - 1}].')

        self.ensure_ready(preload_saes=False)
        assert self.model is not None and self.sae_store is not None
        inputs = self._inputs(text)
        capture: dict = {}
        with self._capture_hook(int(layer), capture):
            self.model(**inputs, use_cache=False)
        hidden = capture['hidden'][0]
        idx = self._resolve_index(int(token_index), hidden.shape[0])
        sae = self.sae_store.get(int(layer))
        encoding = sae.encode(hidden[idx])
        activations = [float(encoding.activation_for(feature_id)) for feature_id in ids]
        directions = torch.stack([sae.decoder_direction(feature_id).float() for feature_id in ids])
        gram = decoder_cosine_matrix(directions)

        rows: list[list[object]] = []
        chart_rows: list[list[object]] = []
        offdiag: list[float] = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                cosine = float(gram[i, j].item())
                offdiag.append(abs(cosine))
                rows.append([ids[i], ids[j], activations[i], activations[j], cosine])
                chart_rows.append([f'{ids[i]} ↔ {ids[j]}', cosine])

        individual_deltas = torch.stack(
            [-float(activation) * direction for activation, direction in zip(activations, directions, strict=True)]
        )
        joint_norm, independent_norm, alignment_ratio = joint_direction_norm_ratio(individual_deltas)
        return FeatureGeometryResult(
            feature_ids=ids,
            layer=int(layer),
            rows=rows,
            chart_rows=chart_rows,
            mean_abs_decoder_cosine=float(sum(offdiag) / len(offdiag)) if offdiag else 0.0,
            max_abs_decoder_cosine=float(max(offdiag)) if offdiag else 0.0,
            joint_ablation_norm=joint_norm,
            independent_norm=independent_norm,
            alignment_ratio=float(alignment_ratio),
        )

    @torch.inference_mode()
    def contrastive_intervention(
        self,
        text: str,
        layer: int,
        token_index: int,
        feature_id: int,
        mode: str,
        coefficient: float,
        target_a: str,
        target_b: str,
    ) -> ContrastiveCausalResult:
        """Measure whether an SAE edit shifts preference between two exact continuations."""
        if not text.strip():
            raise ValueError('Enter a prompt first.')
        if not target_a:
            raise ValueError('Enter preferred continuation A.')
        if not target_b:
            raise ValueError('Enter comparison continuation B.')
        if target_a == target_b:
            raise ValueError('Continuations A and B must be different.')

        self.ensure_ready(preload_saes=False)
        assert self.model is not None and self.tokenizer is not None and self.sae_store is not None
        prompt_inputs = self._inputs(text)
        prompt_len = int(prompt_inputs['input_ids'].shape[1])
        idx = self._resolve_index(int(token_index), prompt_len)
        capture: dict = {}
        with self._capture_hook(int(layer), capture):
            self.model(**prompt_inputs, use_cache=False)
        residual = capture['hidden'][0, idx]
        sae = self.sae_store.get(int(layer))
        encoding = sae.encode(residual)
        activation = float(encoding.activation_for(int(feature_id)))
        delta = residual_delta(
            sae.decoder_direction(int(feature_id)),
            activation,
            InterventionSpec(mode=mode, coefficient=float(coefficient)),
        )
        seed = self._control_seed(text, int(layer), str(int(feature_id)), mode, float(coefficient))
        controls = self._random_controls(delta, seed=seed, count=self.settings.live_random_controls)
        deltas = torch.stack([torch.zeros_like(delta), delta, *controls], dim=0)

        def score(target_text: str):
            target_ids = self._target_ids(target_text)
            scoring_inputs = self._append_target(prompt_inputs, target_ids)
            repeated = self._repeat_inputs(scoring_inputs, deltas.shape[0])
            with self._batch_delta_hook(int(layer), idx, deltas):
                output = self.model(**repeated, use_cache=False)
            summaries = [
                sequence_logprob_summary(output.logits[row], prompt_length=prompt_len, target_ids=target_ids)
                for row in range(output.logits.shape[0])
            ]
            return target_ids, summaries

        ids_a, scores_a = score(target_a)
        ids_b, scores_b = score(target_b)
        base_a_seq, base_a_mean, _ = scores_a[0]
        edit_a_seq, edit_a_mean, _ = scores_a[1]
        base_b_seq, base_b_mean, _ = scores_b[0]
        edit_b_seq, edit_b_mean, _ = scores_b[1]

        baseline_log_odds, modified_log_odds, delta_log_odds = contrastive_log_odds(
            base_a_seq, edit_a_seq, base_b_seq, edit_b_seq
        )
        baseline_norm_pref, modified_norm_pref, delta_norm_pref = contrastive_log_odds(
            base_a_mean, edit_a_mean, base_b_mean, edit_b_mean
        )

        random_delta_log_odds: list[float] = []
        for row in range(2, len(scores_a)):
            random_a_seq = float(scores_a[row][0])
            random_b_seq = float(scores_b[row][0])
            random_delta_log_odds.append(
                float((random_a_seq - random_b_seq) - baseline_log_odds)
            )
        random_signed, random_abs, random_std, empirical_p = self._random_effect_summary(
            random_delta_log_odds, delta_log_odds
        )
        ratio = abs(delta_log_odds) / max(random_abs, 1e-12)
        rows = [
            [
                'A (preferred)',
                target_a,
                len(ids_a),
                float(base_a_seq),
                float(edit_a_seq),
                float(edit_a_seq - base_a_seq),
                float(base_a_mean),
                float(edit_a_mean),
                float(edit_a_mean - base_a_mean),
            ],
            [
                'B (comparison)',
                target_b,
                len(ids_b),
                float(base_b_seq),
                float(edit_b_seq),
                float(edit_b_seq - base_b_seq),
                float(base_b_mean),
                float(edit_b_mean),
                float(edit_b_mean - base_b_mean),
            ],
        ]
        return ContrastiveCausalResult(
            feature_id=int(feature_id),
            layer=int(layer),
            feature_activation=activation,
            perturbation_norm=float(torch.linalg.vector_norm(delta.float()).item()),
            target_a_tokens=[self.tokenizer.decode([int(x)]) for x in ids_a],
            target_b_tokens=[self.tokenizer.decode([int(x)]) for x in ids_b],
            rows=rows,
            baseline_log_odds=baseline_log_odds,
            modified_log_odds=modified_log_odds,
            delta_log_odds=delta_log_odds,
            baseline_normalized_preference=baseline_norm_pref,
            modified_normalized_preference=modified_norm_pref,
            delta_normalized_preference=delta_norm_pref,
            random_signed_mean_delta=random_signed,
            random_abs_mean_delta=random_abs,
            random_delta_std=random_std,
            specificity_ratio=float(ratio),
            empirical_p=float(empirical_p),
            random_control_count=len(controls),
        )

    @torch.inference_mode()
    def concept_contrast_scan(
        self,
        feature_id: int,
        layer: int,
        prompts_per_concept: int | None = None,
    ) -> ConceptContrastResult:
        """Measure one SAE feature using prompt-wide max activation on a balanced concept batch."""
        self.ensure_ready(preload_saes=False)
        assert self.model is not None and self.tokenizer is not None and self.sae_store is not None
        if int(feature_id) < 0 or int(feature_id) >= self.settings.sae_width:
            raise ValueError(f'Feature id must be in [0, {self.settings.sae_width - 1}].')
        n = int(prompts_per_concept or self.settings.contrast_prompts_per_concept)
        if n < 1 or n > 8:
            raise ValueError('Contrast prompts per concept must be between 1 and 8.')
        rows = self._contrast_prompt_rows(n)
        if not rows:
            raise RuntimeError('No controlled contrast prompts are available.')
        texts = [str(row['text']) for row in rows]
        batch = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.settings.max_prompt_tokens,
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        capture: dict = {}
        with self._capture_hook(int(layer), capture):
            self.model(**batch, use_cache=False)
        residuals = capture['hidden']
        sae = self.sae_store.get(int(layer))
        encoding = sae.encode(residuals)
        feature_mask = encoding.indices == int(feature_id)
        token_activations = torch.where(
            feature_mask, encoding.values, torch.zeros_like(encoding.values)
        ).sum(dim=-1)
        attention = batch.get('attention_mask', torch.ones_like(batch['input_ids'])).bool()
        token_activations = torch.where(attention, token_activations, torch.zeros_like(token_activations))
        prompt_activations = token_activations.max(dim=1).values
        values = prompt_activations.detach().float().cpu().tolist()

        grouped: dict[str, list[float]] = {}
        for row, value in zip(rows, values, strict=True):
            grouped.setdefault(str(row['concept']), []).append(float(value))
        table_rows: list[list[object]] = []
        for concept in sorted(grouped):
            vals = grouped[concept]
            tensor = torch.tensor(vals, dtype=torch.float64)
            active = [value for value in vals if value > 0]
            positive_mean = float(sum(active) / len(active)) if active else 0.0
            table_rows.append(
                [
                    concept,
                    len(vals),
                    float(tensor.mean().item()),
                    float(torch.median(tensor).item()),
                    float(len(active) / len(vals)),
                    positive_mean,
                    float(max(vals) if vals else 0.0),
                ]
            )
        table_rows.sort(key=lambda row: (float(row[2]), float(row[4]), float(row[6])), reverse=True)
        active_prompt_count = sum(1 for value in values if float(value) > 0)
        if table_rows and float(table_rows[0][2]) > 0:
            leader: str | None = str(table_rows[0][0])
            first = float(table_rows[0][2])
            second = float(table_rows[1][2]) if len(table_rows) > 1 else 0.0
            ratio: float | None = first / second if second > 0 else None
        else:
            leader = None
            ratio = None
        chart_rows = [[str(row[0]), float(row[2])] for row in table_rows]
        return ConceptContrastResult(
            feature_id=int(feature_id),
            layer=int(layer),
            prompts_per_concept=n,
            rows=table_rows,
            chart_rows=chart_rows,
            leading_concept=leader,
            leading_ratio=ratio,
            active_prompt_count=int(active_prompt_count),
            total_prompt_count=len(values),
        )

    @torch.inference_mode()
    def concept_feature_discovery(
        self,
        concept: str,
        layer: int,
        prompts_per_concept: int | None = None,
        top_n: int = 12,
        ranking_mode: str = 'balanced_selectivity',
        current_text: str | None = None,
        current_token_index: int = -1,
    ) -> ConceptFeatureDiscoveryResult:
        """Find exploratory concept candidates and show whether they are usable in the current Workbench context.

        ``balanced_selectivity`` downweights globally high-activation features by combining target selectivity,
        target coverage, and target activation magnitude. ``raw_mean_difference`` preserves the simpler raw
        mean-difference ranking for comparison. ``causal_ready`` further requires activation at the currently
        selected Workbench token and ranks those compatible candidates by balanced evidence plus a log-scaled
        current-token activation term. If a Workbench prompt is supplied it is appended to the same model batch,
        so current-prompt compatibility does not require another forward pass.
        """
        self.ensure_ready(preload_saes=False)
        assert self.model is not None and self.tokenizer is not None and self.sae_store is not None
        concept = str(concept).strip()
        if not concept:
            raise ValueError('Choose a target concept.')
        n = int(prompts_per_concept or self.settings.contrast_prompts_per_concept)
        if n < 1 or n > 8:
            raise ValueError('Discovery prompts per concept must be between 1 and 8.')
        top_n = int(top_n)
        if top_n < 1 or top_n > 25:
            raise ValueError('Number of candidate features must be between 1 and 25.')
        if ranking_mode not in {'balanced_selectivity', 'raw_mean_difference', 'causal_ready'}:
            raise ValueError(
                "ranking_mode must be 'balanced_selectivity', 'raw_mean_difference', or 'causal_ready'."
            )

        rows = self._contrast_prompt_rows(n)
        available = sorted({str(row['concept']) for row in rows})
        if concept not in available:
            raise ValueError(f'Concept must be one of {available}.')

        current_text = str(current_text or '').strip()
        current_context_available = bool(current_text)
        texts = [str(row['text']) for row in rows]
        if current_context_available:
            texts.append(current_text)

        batch = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.settings.max_prompt_tokens,
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        capture: dict = {}
        with self._capture_hook(int(layer), capture):
            self.model(**batch, use_cache=False)
        sae = self.sae_store.get(int(layer))
        encoding = sae.encode(capture['hidden'])
        attention = batch.get('attention_mask', torch.ones_like(batch['input_ids'])).bool()
        valid = attention.unsqueeze(-1).expand_as(encoding.values)
        values = torch.where(valid, encoding.values, torch.zeros_like(encoding.values)).float()

        dense = torch.zeros(
            (values.shape[0], self.settings.sae_width),
            device=values.device,
            dtype=torch.float32,
        )
        dense.scatter_reduce_(
            1,
            encoding.indices.reshape(values.shape[0], -1),
            values.reshape(values.shape[0], -1),
            reduce='amax',
            include_self=True,
        )

        controlled_count = len(rows)
        controlled_dense = dense[:controlled_count]
        target_mask = torch.tensor(
            [str(row['concept']) == concept for row in rows],
            device=dense.device,
            dtype=torch.bool,
        )
        other_mask = ~target_mask
        target = controlled_dense[target_mask]
        other = controlled_dense[other_mask]
        target_mean = target.mean(dim=0)
        other_mean = other.mean(dim=0)
        target_rate = (target > 0).float().mean(dim=0)
        other_rate = (other > 0).float().mean(dim=0)
        mean_diff = target_mean - other_mean
        selectivity = mean_diff / (target_mean + other_mean + 1e-8)

        # A scale-aware but selectivity-first exploratory score. log1p prevents very large SAE coefficients from
        # overwhelming features that are much more exclusive to the target concept.
        balanced_score = selectivity.clamp_min(0) * target_rate * torch.log1p(target_mean.clamp_min(0))

        current_prompt_max = torch.zeros(self.settings.sae_width, device=dense.device, dtype=torch.float32)
        current_token_dense = torch.zeros_like(current_prompt_max)
        resolved_current_idx: int | None = None
        if current_context_available:
            current_row = controlled_count
            current_prompt_max = dense[current_row]
            valid_positions = torch.nonzero(attention[current_row], as_tuple=False).flatten()
            prompt_len = int(valid_positions.numel())
            if prompt_len:
                resolved_current_idx = self._resolve_index(int(current_token_index), prompt_len)
                padded_position = int(valid_positions[resolved_current_idx].item())
                token_ids = encoding.indices[current_row, padded_position]
                token_values = encoding.values[current_row, padded_position].float().clamp_min(0)
                current_token_dense.scatter_reduce_(
                    0,
                    token_ids,
                    token_values,
                    reduce='amax',
                    include_self=True,
                )

        # Stability diagnostics reuse the same controlled activation batch, so they add no model inference.
        # Split-half overlap is intentionally simple; deterministic balanced bootstrap support gives a second
        # view of how often each displayed feature survives small changes to the live prompt sample.
        split_half_k: int | None = None
        split_half_shared_count = 0
        split_half_jaccard: float | None = None
        resample_replicates = 0
        resample_rank_lists: list[list[int]] = []
        row_concepts = [str(row['concept']) for row in rows]

        def _rank_indices(row_indices: list[int]) -> list[int]:
            if not row_indices:
                return []
            index_tensor = torch.tensor(row_indices, device=dense.device, dtype=torch.long)
            sub_dense = controlled_dense.index_select(0, index_tensor)
            sub_labels = [row_concepts[index] for index in row_indices]
            sub_target_mask = torch.tensor(
                [label == concept for label in sub_labels],
                device=dense.device,
                dtype=torch.bool,
            )
            sub_other_mask = ~sub_target_mask
            sub_target = sub_dense[sub_target_mask]
            sub_other = sub_dense[sub_other_mask]
            if sub_target.shape[0] == 0 or sub_other.shape[0] == 0:
                return []
            sub_target_mean = sub_target.mean(dim=0)
            sub_other_mean = sub_other.mean(dim=0)
            sub_target_rate = (sub_target > 0).float().mean(dim=0)
            sub_mean_diff = sub_target_mean - sub_other_mean
            sub_selectivity = sub_mean_diff / (sub_target_mean + sub_other_mean + 1e-8)
            sub_balanced = (
                sub_selectivity.clamp_min(0)
                * sub_target_rate
                * torch.log1p(sub_target_mean.clamp_min(0))
            )
            sub_eligible = (sub_target_mean > 0) & (sub_mean_diff > 0)
            if ranking_mode == 'causal_ready':
                sub_eligible = sub_eligible & (current_token_dense > 0)
                sub_ranking = sub_balanced * torch.log1p(current_token_dense.clamp_min(0))
            elif ranking_mode == 'balanced_selectivity':
                sub_ranking = sub_balanced
            else:
                sub_ranking = sub_mean_diff
            sub_idx = torch.nonzero(sub_eligible, as_tuple=False).flatten()
            if sub_idx.numel() == 0:
                return []
            sub_order = torch.argsort(sub_ranking[sub_idx], descending=True)
            return [int(value.item()) for value in sub_idx[sub_order[:top_n]]]

        if n >= 2:
            half = max(1, n // 2)
            seen_by_concept: dict[str, int] = {}
            indices_a: list[int] = []
            indices_b: list[int] = []
            for row_idx, row_concept in enumerate(row_concepts):
                local_idx = seen_by_concept.get(row_concept, 0)
                seen_by_concept[row_concept] = local_idx + 1
                if local_idx < half:
                    indices_a.append(row_idx)
                else:
                    indices_b.append(row_idx)
            ids_a = _rank_indices(indices_a)
            ids_b = _rank_indices(indices_b)
            if ids_a and ids_b:
                set_a, set_b = set(ids_a), set(ids_b)
                shared = set_a & set_b
                union = set_a | set_b
                split_half_k = min(len(ids_a), len(ids_b), top_n)
                split_half_shared_count = len(shared)
                split_half_jaccard = float(len(shared) / len(union)) if union else 1.0

            pools: dict[str, list[int]] = {}
            for row_idx, row_concept in enumerate(row_concepts):
                pools.setdefault(row_concept, []).append(row_idx)
            seed = 13013 + int(layer) * 97 + n * 17 + sum(ord(ch) for ch in concept)
            rng = random.Random(seed)
            resample_replicates = 32
            for _ in range(resample_replicates):
                sampled_indices: list[int] = []
                for row_concept in sorted(pools):
                    pool = pools[row_concept]
                    sampled_indices.extend(rng.choice(pool) for _ in range(len(pool)))
                ranked = _rank_indices(sampled_indices)
                if ranked:
                    resample_rank_lists.append(ranked)
            resample_replicates = len(resample_rank_lists)

        eligible = (target_mean > 0) & (mean_diff > 0)
        if ranking_mode == 'causal_ready':
            if not current_context_available or resolved_current_idx is None:
                raise ValueError(
                    "Causal-ready ranking requires a current Workbench prompt/token. "
                    "Set the Workbench prompt/layer/token first."
                )
            eligible = eligible & (current_token_dense > 0)
            ranking_values = balanced_score * torch.log1p(current_token_dense.clamp_min(0))
        elif ranking_mode == 'balanced_selectivity':
            ranking_values = balanced_score
        else:
            ranking_values = mean_diff

        candidate_idx = torch.nonzero(eligible, as_tuple=False).flatten()
        if candidate_idx.numel() == 0:
            return ConceptFeatureDiscoveryResult(
                concept=concept,
                layer=int(layer),
                prompts_per_concept=n,
                top_n=top_n,
                ranking_mode=ranking_mode,
                rows=[],
                chart_rows=[],
                candidate_ids=[],
                default_candidate_id=None,
                current_context_available=current_context_available,
                current_token_index=resolved_current_idx,
                displayed_current_active_count=0,
                split_half_k=split_half_k,
                split_half_shared_count=split_half_shared_count,
                split_half_jaccard=split_half_jaccard,
                resample_replicates=resample_replicates,
                resample_mean_support=None,
                resample_high_support_count=0,
            )

        order = torch.argsort(ranking_values[candidate_idx], descending=True)
        candidate_idx = candidate_idx[order[:top_n]]

        resample_support: dict[int, float] = {}
        resample_median_rank: dict[int, float | None] = {}
        displayed_supports: list[float] = []
        if resample_replicates:
            for feature_tensor in candidate_idx:
                fid = int(feature_tensor.item())
                ranks = [ranked.index(fid) + 1 for ranked in resample_rank_lists if fid in ranked]
                support = len(ranks) / resample_replicates
                if ranks:
                    ordered_ranks = sorted(ranks)
                    midpoint = len(ordered_ranks) // 2
                    if len(ordered_ranks) % 2:
                        median_rank = float(ordered_ranks[midpoint])
                    else:
                        median_rank = float((ordered_ranks[midpoint - 1] + ordered_ranks[midpoint]) / 2)
                else:
                    median_rank = None
                resample_support[fid] = float(support)
                resample_median_rank[fid] = median_rank
                displayed_supports.append(float(support))
        resample_mean_support = (
            float(sum(displayed_supports) / len(displayed_supports)) if displayed_supports else None
        )
        resample_high_support_count = sum(support >= 0.75 for support in displayed_supports)

        table_rows: list[list[object]] = []
        chart_rows: list[list[object]] = []
        default_candidate_id: int | None = None
        for rank, feature_tensor in enumerate(candidate_idx, start=1):
            fid = int(feature_tensor.item())
            current_max = float(current_prompt_max[fid].item()) if current_context_available else 0.0
            current_token = float(current_token_dense[fid].item()) if current_context_available else 0.0
            if default_candidate_id is None and current_token > 0:
                default_candidate_id = fid
            score = float(ranking_values[fid].item())
            row = [
                rank,
                fid,
                score,
                float(target_mean[fid].item()),
                float(other_mean[fid].item()),
                float(mean_diff[fid].item()),
                float(selectivity[fid].item()),
                float(target_rate[fid].item()),
                float(other_rate[fid].item()),
                current_max,
                current_token,
                bool(current_token > 0),
                resample_support.get(fid) if resample_replicates else None,
                resample_median_rank.get(fid) if resample_replicates else None,
            ]
            table_rows.append(row)
            chart_rows.append([str(fid), score])

        if default_candidate_id is None and candidate_idx.numel():
            default_candidate_id = int(candidate_idx[0].item())

        return ConceptFeatureDiscoveryResult(
            concept=concept,
            layer=int(layer),
            prompts_per_concept=n,
            top_n=top_n,
            ranking_mode=ranking_mode,
            rows=table_rows,
            chart_rows=chart_rows,
            candidate_ids=[int(x.item()) for x in candidate_idx],
            default_candidate_id=default_candidate_id,
            current_context_available=current_context_available,
            current_token_index=resolved_current_idx,
            displayed_current_active_count=sum(bool(row[11]) for row in table_rows),
            split_half_k=split_half_k,
            split_half_shared_count=split_half_shared_count,
            split_half_jaccard=split_half_jaccard,
            resample_replicates=resample_replicates,
            resample_mean_support=resample_mean_support,
            resample_high_support_count=resample_high_support_count,
        )

    @torch.inference_mode()
    def candidate_causal_screen(
        self,
        text: str,
        layer: int,
        token_index: int,
        feature_ids: Sequence[int],
        target_text: str,
    ) -> CandidateCausalScreenResult:
        """Cheaply triage several candidate features with one batched ablation screen.

        This deliberately omits random controls. Its purpose is to rank candidates before
        spending a full live causal test (with the random-control ensemble) on one or two
        promising features. All feature ablations share the same batched zero-edit reference.
        """
        if not text.strip():
            raise ValueError('Enter and inspect a Workbench prompt first.')
        if not target_text.strip():
            raise ValueError('Enter a target continuation for candidate causal screening.')
        ids = list(dict.fromkeys(int(x) for x in feature_ids))
        if not ids:
            raise ValueError('Select at least one candidate feature to screen.')
        if len(ids) > 8:
            raise ValueError('Candidate causal screening supports at most eight features per run.')
        if any(feature_id < 0 or feature_id >= self.settings.sae_width for feature_id in ids):
            raise ValueError(f'Feature ids must be in [0, {self.settings.sae_width - 1}].')

        self.ensure_ready(preload_saes=False)
        assert self.model is not None and self.tokenizer is not None and self.sae_store is not None
        prompt_inputs = self._inputs(text)
        prompt_len = int(prompt_inputs['input_ids'].shape[1])
        idx = self._resolve_index(int(token_index), prompt_len)
        target_ids = self._target_ids(target_text)
        full_inputs = self._append_target(prompt_inputs, target_ids)
        sae = self.sae_store.get(int(layer))

        capture: dict = {}
        with self._capture_hook(int(layer), capture):
            single_baseline_out = self.model(**full_inputs, use_cache=False)
        single_logits = single_baseline_out.logits[0]
        _, single_mean, _ = sequence_logprob_summary(
            single_logits, prompt_length=prompt_len, target_ids=target_ids
        )

        residual = capture['hidden'][0, idx]
        encoding = sae.encode(residual)
        activations = [float(encoding.activation_for(feature_id)) for feature_id in ids]
        deltas = [
            residual_delta(
                sae.decoder_direction(feature_id),
                activation,
                InterventionSpec('ablate', 0.0),
            )
            for feature_id, activation in zip(ids, activations, strict=True)
        ]
        all_deltas = torch.stack([torch.zeros_like(deltas[0]), *deltas], dim=0)
        repeated = self._repeat_inputs(full_inputs, all_deltas.shape[0])
        with self._batch_delta_hook(int(layer), idx, all_deltas):
            outputs = self.model(**repeated, use_cache=False)

        baseline_logits = outputs.logits[0]
        baseline_seq, baseline_mean, _ = sequence_logprob_summary(
            baseline_logits, prompt_length=prompt_len, target_ids=target_ids
        )
        baseline_next = baseline_logits[prompt_len - 1]
        execution_drift_mean = float(baseline_mean - single_mean)
        execution_drift_js = js_divergence_from_logits(
            single_logits[prompt_len - 1], baseline_next
        )

        scored: list[dict[str, object]] = []
        for row_idx, (feature_id, activation, delta) in enumerate(
            zip(ids, activations, deltas, strict=True), start=1
        ):
            logits = outputs.logits[row_idx]
            seq_logp, mean_logp, _ = sequence_logprob_summary(
                logits, prompt_length=prompt_len, target_ids=target_ids
            )
            mean_delta = float(mean_logp - baseline_mean)
            seq_delta = float(seq_logp - baseline_seq)
            js = float(js_divergence_from_logits(baseline_next, logits[prompt_len - 1]))
            norm = float(torch.linalg.vector_norm(delta.float()).item())
            scored.append(
                {
                    'feature_id': int(feature_id),
                    'activation': float(activation),
                    'active': bool(activation > 0),
                    'norm': norm,
                    'mean_delta': mean_delta,
                    'seq_delta': seq_delta,
                    'js': js,
                }
            )

        scored.sort(
            key=lambda item: (abs(float(item['mean_delta'])), float(item['js'])),
            reverse=True,
        )
        rows: list[list[object]] = []
        chart_rows: list[list[object]] = []
        for rank, item in enumerate(scored, start=1):
            feature_id = int(item['feature_id'])
            mean_delta = float(item['mean_delta'])
            rows.append(
                [
                    rank,
                    feature_id,
                    float(item['activation']),
                    bool(item['active']),
                    float(item['norm']),
                    mean_delta,
                    float(item['seq_delta']),
                    float(item['js']),
                ]
            )
            chart_rows.append([str(feature_id), mean_delta])

        return CandidateCausalScreenResult(
            feature_ids=[int(item['feature_id']) for item in scored],
            target_tokens=[self.tokenizer.decode([int(token_id)]) for token_id in target_ids],
            rows=rows,
            chart_rows=chart_rows,
            active_feature_count=sum(bool(item['active']) for item in scored),
            candidate_count=len(scored),
            execution_drift_mean_logprob=execution_drift_mean,
            execution_drift_js=float(execution_drift_js),
        )

    @torch.inference_mode()
    def candidate_specificity_screen(
        self,
        text: str,
        layer: int,
        token_index: int,
        feature_ids: Sequence[int],
        target_text: str,
    ) -> CandidateSpecificityResult:
        """Compare a small candidate set against per-feature norm-matched random ensembles.

        This is the controlled follow-up to :meth:`candidate_causal_screen`. Each SAE
        ablation gets its own deterministic random-control ensemble with the same L2 norm.
        All targeted and random conditions share one batched zero-edit reference so the
        comparison is both GPU-efficient and execution-context consistent.
        """
        if not text.strip():
            raise ValueError('Enter and inspect a Workbench prompt first.')
        if not target_text.strip():
            raise ValueError('Enter a target continuation for controlled candidate comparison.')
        ids = list(dict.fromkeys(int(x) for x in feature_ids))
        if not ids:
            raise ValueError('Select at least one candidate feature for controlled comparison.')
        if len(ids) > 3:
            raise ValueError('Controlled candidate comparison supports at most three features per run.')
        if any(feature_id < 0 or feature_id >= self.settings.sae_width for feature_id in ids):
            raise ValueError(f'Feature ids must be in [0, {self.settings.sae_width - 1}].')

        self.ensure_ready(preload_saes=False)
        assert self.model is not None and self.tokenizer is not None and self.sae_store is not None
        prompt_inputs = self._inputs(text)
        prompt_len = int(prompt_inputs['input_ids'].shape[1])
        idx = self._resolve_index(int(token_index), prompt_len)
        target_ids = self._target_ids(target_text)
        full_inputs = self._append_target(prompt_inputs, target_ids)
        sae = self.sae_store.get(int(layer))

        capture: dict = {}
        with self._capture_hook(int(layer), capture):
            single_baseline_out = self.model(**full_inputs, use_cache=False)
        single_logits = single_baseline_out.logits[0]
        _, single_mean, _ = sequence_logprob_summary(
            single_logits, prompt_length=prompt_len, target_ids=target_ids
        )

        residual = capture['hidden'][0, idx]
        encoding = sae.encode(residual)
        activations = [float(encoding.activation_for(feature_id)) for feature_id in ids]

        zero = torch.zeros_like(residual)
        all_deltas: list[torch.Tensor] = [zero]
        metadata: list[tuple[int, str, float, float]] = []
        for feature_id, activation in zip(ids, activations, strict=True):
            delta = residual_delta(
                sae.decoder_direction(feature_id),
                activation,
                InterventionSpec('ablate', 0.0),
            )
            norm = float(torch.linalg.vector_norm(delta.float()).item())
            all_deltas.append(delta)
            metadata.append((feature_id, 'sae', activation, norm))
            seed = self._control_seed(text, int(layer), str(feature_id), 'candidate_specificity', 0.0)
            controls = self._random_controls(
                delta, seed=seed, count=self.settings.live_random_controls
            )
            for control_idx, control in enumerate(controls):
                all_deltas.append(control)
                metadata.append((feature_id, f'random_{control_idx}', activation, norm))

        delta_batch = torch.stack(all_deltas, dim=0)
        repeated = self._repeat_inputs(full_inputs, delta_batch.shape[0])
        with self._batch_delta_hook(int(layer), idx, delta_batch):
            outputs = self.model(**repeated, use_cache=False)

        baseline_logits = outputs.logits[0]
        baseline_seq, baseline_mean, _ = sequence_logprob_summary(
            baseline_logits, prompt_length=prompt_len, target_ids=target_ids
        )
        baseline_next = baseline_logits[prompt_len - 1]
        execution_drift_mean = float(baseline_mean - single_mean)
        execution_drift_js = js_divergence_from_logits(
            single_logits[prompt_len - 1], baseline_next
        )

        grouped: dict[int, dict[str, object]] = {
            feature_id: {
                'activation': activation,
                'norm': 0.0,
                'sae': None,
                'random': [],
            }
            for feature_id, activation in zip(ids, activations, strict=True)
        }

        for output_idx, meta in enumerate(metadata, start=1):
            feature_id, kind, activation, norm = meta
            logits = outputs.logits[output_idx]
            seq_logp, mean_logp, _ = sequence_logprob_summary(
                logits, prompt_length=prompt_len, target_ids=target_ids
            )
            item = {
                'seq_delta': float(seq_logp - baseline_seq),
                'mean_delta': float(mean_logp - baseline_mean),
                'js': float(js_divergence_from_logits(baseline_next, logits[prompt_len - 1])),
            }
            grouped[feature_id]['activation'] = float(activation)
            grouped[feature_id]['norm'] = float(norm)
            if kind == 'sae':
                grouped[feature_id]['sae'] = item
            else:
                random_items = grouped[feature_id]['random']
                assert isinstance(random_items, list)
                random_items.append(item)

        scored: list[dict[str, object]] = []
        for feature_id in ids:
            group = grouped[feature_id]
            sae_item = group['sae']
            random_items = group['random']
            assert isinstance(sae_item, dict)
            assert isinstance(random_items, list) and random_items
            mean_delta = float(sae_item['mean_delta'])
            js = float(sae_item['js'])
            random_mean_deltas = [float(item['mean_delta']) for item in random_items]
            random_js_values = [float(item['js']) for item in random_items]
            random_signed, random_abs, random_std, target_p = self._random_effect_summary(
                random_mean_deltas, mean_delta
            )
            _, random_js_abs, random_js_std, js_p = self._random_effect_summary(
                random_js_values, js
            )
            scored.append(
                {
                    'feature_id': int(feature_id),
                    'activation': float(group['activation']),
                    'active': bool(float(group['activation']) > 0),
                    'norm': float(group['norm']),
                    'mean_delta': mean_delta,
                    'seq_delta': float(sae_item['seq_delta']),
                    'random_signed': float(random_signed),
                    'random_abs': float(random_abs),
                    'random_std': float(random_std),
                    'target_ratio': float(abs(mean_delta) / max(random_abs, 1e-12)),
                    'target_p': float(target_p),
                    'js': js,
                    'random_js': float(random_js_abs),
                    'random_js_std': float(random_js_std),
                    'js_ratio': float(js / max(random_js_abs, 1e-12)),
                    'js_p': float(js_p),
                }
            )

        scored.sort(
            key=lambda item: (float(item['target_ratio']), abs(float(item['mean_delta']))),
            reverse=True,
        )
        rows: list[list[object]] = []
        chart_rows: list[list[object]] = []
        for rank, item in enumerate(scored, start=1):
            feature_id = int(item['feature_id'])
            rows.append(
                [
                    rank,
                    feature_id,
                    float(item['activation']),
                    bool(item['active']),
                    float(item['norm']),
                    float(item['mean_delta']),
                    float(item['random_signed']),
                    float(item['random_abs']),
                    float(item['random_std']),
                    float(item['target_ratio']),
                    float(item['target_p']),
                    float(item['seq_delta']),
                    float(item['js']),
                    float(item['random_js']),
                    float(item['random_js_std']),
                    float(item['js_ratio']),
                    float(item['js_p']),
                ]
            )
            chart_rows.extend(
                [
                    [str(feature_id), 'Target specificity', float(item['target_ratio'])],
                    [str(feature_id), 'JS specificity', float(item['js_ratio'])],
                ]
            )

        return CandidateSpecificityResult(
            feature_ids=[int(item['feature_id']) for item in scored],
            target_tokens=[self.tokenizer.decode([int(token_id)]) for token_id in target_ids],
            rows=rows,
            chart_rows=chart_rows,
            active_feature_count=sum(bool(item['active']) for item in scored),
            candidate_count=len(scored),
            random_control_count=self.settings.live_random_controls,
            execution_drift_mean_logprob=execution_drift_mean,
            execution_drift_js=float(execution_drift_js),
        )

    @torch.inference_mode()
    def candidate_cross_target_profile(
        self,
        text: str,
        layer: int,
        token_index: int,
        feature_ids: Sequence[int],
        targets: Sequence[str],
    ) -> CandidateCrossTargetResult:
        """Profile native candidate ablations across several exact target continuations.

        This is a screening diagnostic, not a random-controlled causal claim. The residual
        representation and native feature deltas are captured once from the Workbench prompt.
        Each target is then evaluated with one small batched forward containing the zero-edit
        reference plus every selected candidate ablation.
        """
        if not text.strip():
            raise ValueError('Enter and inspect a Workbench prompt first.')
        ids = list(dict.fromkeys(int(x) for x in feature_ids))
        if not ids:
            raise ValueError('Select at least one candidate feature for cross-target profiling.')
        if len(ids) > 3:
            raise ValueError('Cross-target profiling supports at most three features per run.')
        if any(feature_id < 0 or feature_id >= self.settings.sae_width for feature_id in ids):
            raise ValueError(f'Feature ids must be in [0, {self.settings.sae_width - 1}].')

        target_list: list[str] = []
        seen_targets: set[str] = set()
        for raw in targets:
            target = str(raw)
            if not target.strip() or target in seen_targets:
                continue
            seen_targets.add(target)
            target_list.append(target)
        if len(target_list) < 2:
            raise ValueError('Enter at least two distinct target continuations.')
        if len(target_list) > 5:
            raise ValueError('Cross-target profiling supports at most five target continuations.')

        self.ensure_ready(preload_saes=False)
        assert self.model is not None and self.tokenizer is not None and self.sae_store is not None
        prompt_inputs = self._inputs(text)
        prompt_len = int(prompt_inputs['input_ids'].shape[1])
        idx = self._resolve_index(int(token_index), prompt_len)
        sae = self.sae_store.get(int(layer))

        capture: dict = {}
        with self._capture_hook(int(layer), capture):
            self.model(**prompt_inputs, use_cache=False)
        residual = capture['hidden'][0, idx]
        encoding = sae.encode(residual)
        activations = [float(encoding.activation_for(feature_id)) for feature_id in ids]
        deltas = [
            residual_delta(
                sae.decoder_direction(feature_id),
                activation,
                InterventionSpec('ablate', 0.0),
            )
            for feature_id, activation in zip(ids, activations, strict=True)
        ]
        delta_batch = torch.stack([torch.zeros_like(residual), *deltas], dim=0)

        rows: list[list[object]] = []
        chart_rows: list[list[object]] = []
        by_feature: dict[int, list[tuple[str, float, float]]] = {feature_id: [] for feature_id in ids}

        for target_text in target_list:
            target_ids = self._target_ids(target_text)
            full_inputs = self._append_target(prompt_inputs, target_ids)
            repeated = self._repeat_inputs(full_inputs, delta_batch.shape[0])
            with self._batch_delta_hook(int(layer), idx, delta_batch):
                outputs = self.model(**repeated, use_cache=False)

            baseline_logits = outputs.logits[0]
            baseline_seq, baseline_mean, _ = sequence_logprob_summary(
                baseline_logits, prompt_length=prompt_len, target_ids=target_ids
            )
            baseline_next = baseline_logits[prompt_len - 1]
            token_count = len(target_ids)
            for output_idx, (feature_id, activation, delta) in enumerate(
                zip(ids, activations, deltas, strict=True), start=1
            ):
                logits = outputs.logits[output_idx]
                seq_logp, mean_logp, _ = sequence_logprob_summary(
                    logits, prompt_length=prompt_len, target_ids=target_ids
                )
                mean_delta = float(mean_logp - baseline_mean)
                seq_delta = float(seq_logp - baseline_seq)
                js = float(js_divergence_from_logits(baseline_next, logits[prompt_len - 1]))
                rows.append(
                    [
                        int(feature_id),
                        target_text,
                        int(token_count),
                        float(activation),
                        float(torch.linalg.vector_norm(delta.float()).item()),
                        mean_delta,
                        seq_delta,
                        js,
                    ]
                )
                chart_rows.append([target_text, str(feature_id), mean_delta])
                by_feature[feature_id].append((target_text, mean_delta, js))

        summary_rows: list[list[object]] = []
        pairwise_rows: list[list[object]] = []
        for feature_id in ids:
            items = by_feature[feature_id]
            strongest = max(items, key=lambda item: abs(item[1]))
            strongest_abs = abs(float(strongest[1]))
            other_abs = [abs(float(item[1])) for item in items if item is not strongest]
            mean_other = float(sum(other_abs) / len(other_abs)) if other_abs else 0.0
            profile_ratio = float(strongest_abs / max(mean_other, 1e-12))
            deltas = [float(item[1]) for item in items]
            abs_values = [abs(delta) for delta in deltas]
            total_abs = float(sum(abs_values))
            if total_abs > 0 and len(abs_values) > 1:
                proportions = [value / total_abs for value in abs_values if value > 0]
                normalized_entropy = float(
                    -sum(value * math.log(value) for value in proportions) / math.log(len(abs_values))
                )
            else:
                normalized_entropy = 0.0
            effect_concentration = float(1.0 - normalized_entropy)
            signed_bias = float(sum(deltas) / total_abs) if total_abs > 0 else 0.0
            signs = {1 if delta > 0 else -1 if delta < 0 else 0 for delta in deltas}
            nonzero_signs = {sign for sign in signs if sign != 0}
            sign_consistency = 'same sign' if len(nonzero_signs) <= 1 else 'mixed signs'
            if effect_concentration >= 0.25:
                profile_pattern = (
                    'target-concentrated / mixed-sign'
                    if sign_consistency == 'mixed signs'
                    else 'target-concentrated / same-sign'
                )
            elif signed_bias <= -0.8:
                profile_pattern = 'broad same-sign suppression'
            elif signed_bias >= 0.8:
                profile_pattern = 'broad same-sign enhancement'
            else:
                profile_pattern = 'broad mixed-sign'
            summary_rows.append(
                [
                    int(feature_id),
                    str(strongest[0]),
                    float(strongest[1]),
                    strongest_abs,
                    mean_other,
                    profile_ratio,
                    sign_consistency,
                    normalized_entropy,
                    effect_concentration,
                    signed_bias,
                    profile_pattern,
                    max(float(item[2]) for item in items),
                ]
            )
            for left_idx in range(len(items)):
                for right_idx in range(left_idx + 1, len(items)):
                    target_a, delta_a, _ = items[left_idx]
                    target_b, delta_b, _ = items[right_idx]
                    preference_shift = float(delta_a - delta_b)
                    direction = (
                        f'toward {target_a}'
                        if preference_shift > 0
                        else f'toward {target_b}' if preference_shift < 0 else 'no shift'
                    )
                    pairwise_rows.append(
                        [
                            int(feature_id),
                            str(target_a),
                            str(target_b),
                            preference_shift,
                            abs(preference_shift),
                            direction,
                        ]
                    )
        summary_rows.sort(key=lambda row: float(row[3]), reverse=True)
        pairwise_rows.sort(key=lambda row: float(row[4]), reverse=True)

        return CandidateCrossTargetResult(
            feature_ids=ids,
            targets=target_list,
            rows=rows,
            chart_rows=chart_rows,
            summary_rows=summary_rows,
            pairwise_rows=pairwise_rows,
            active_feature_count=sum(activation > 0 for activation in activations),
        )

    @staticmethod
    def _cue_prompt(stem: str, cue: str) -> str:
        stem = stem.rstrip()
        cue = cue.strip()
        if not cue:
            return stem
        if cue[0] in ':;,.!?=)]}':
            return stem + cue
        return stem + ' ' + cue

    @torch.inference_mode()
    def feature_cue_scan(
        self,
        feature_id: int,
        layer: int,
        prompt_stem: str,
        cues: Sequence[str],
    ) -> FeatureCueScanResult:
        """Measure one feature at the final token after appending controlled completion cues."""
        if not prompt_stem.strip():
            raise ValueError('Enter a prompt stem.')
        cue_list: list[str] = []
        seen: set[str] = set()
        for raw in cues:
            cue = str(raw).strip()
            if not cue or cue in seen:
                continue
            seen.add(cue)
            cue_list.append(cue)
        if not cue_list:
            raise ValueError('Enter at least one cue.')
        if len(cue_list) > 12:
            raise ValueError('Cue scan supports at most 12 cues per run.')
        if int(feature_id) < 0 or int(feature_id) >= self.settings.sae_width:
            raise ValueError(f'Feature id must be in [0, {self.settings.sae_width - 1}].')

        self.ensure_ready(preload_saes=False)
        assert self.model is not None and self.tokenizer is not None and self.sae_store is not None
        prompts = [self._cue_prompt(prompt_stem, cue) for cue in cue_list]
        batch = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.settings.max_prompt_tokens,
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        capture: dict = {}
        with self._capture_hook(int(layer), capture):
            self.model(**batch, use_cache=False)
        sae = self.sae_store.get(int(layer))
        encoding = sae.encode(capture['hidden'])
        attention = batch.get('attention_mask', torch.ones_like(batch['input_ids'])).bool()

        rows: list[list[object]] = []
        chart_rows: list[list[object]] = []
        active_count = 0
        for row_idx, (cue, full_prompt) in enumerate(zip(cue_list, prompts, strict=True)):
            valid_positions = torch.nonzero(attention[row_idx], as_tuple=False).flatten()
            final_pos = int(valid_positions[-1].item())
            indices = encoding.indices[row_idx, final_pos]
            values = encoding.values[row_idx, final_pos]
            mask = indices == int(feature_id)
            activation = float(values[mask][0].item()) if bool(mask.any()) else 0.0
            active = activation > 0
            active_count += int(active)
            token_id = int(batch['input_ids'][row_idx, final_pos].item())
            final_token = self.tokenizer.decode([token_id])
            rows.append([cue, full_prompt, repr(final_token), activation, active])
            chart_rows.append([cue, activation])

        return FeatureCueScanResult(
            feature_id=int(feature_id),
            layer=int(layer),
            prompt_stem=prompt_stem,
            rows=rows,
            chart_rows=chart_rows,
            active_cue_count=active_count,
            cue_count=len(cue_list),
        )

    @torch.inference_mode()
    def feature_cue_context_scan(
        self,
        feature_id: int,
        layer: int,
        stems: Sequence[str],
        cues: Sequence[str],
    ) -> FeatureCueContextResult:
        """Cross completion cues with multiple prompt stems in one batch.

        This distinguishes a cue-specific response (for example a feature that responds to ``is`` everywhere)
        from a context-sensitive completion-boundary response.
        """
        stem_list: list[str] = []
        seen_stems: set[str] = set()
        for raw in stems:
            stem = str(raw).strip()
            if not stem or stem in seen_stems:
                continue
            seen_stems.add(stem)
            stem_list.append(stem)
        cue_list: list[str] = []
        seen_cues: set[str] = set()
        for raw in cues:
            cue = str(raw).strip()
            if not cue or cue in seen_cues:
                continue
            seen_cues.add(cue)
            cue_list.append(cue)
        if not stem_list:
            raise ValueError('Enter at least one prompt stem.')
        if not cue_list:
            raise ValueError('Enter at least one completion cue.')
        if len(stem_list) > 8:
            raise ValueError('Cue-context scan supports at most 8 prompt stems.')
        if len(cue_list) > 8:
            raise ValueError('Cue-context scan supports at most 8 completion cues.')
        if len(stem_list) * len(cue_list) > 40:
            raise ValueError('Cue-context scan supports at most 40 stem × cue conditions per run.')
        if int(feature_id) < 0 or int(feature_id) >= self.settings.sae_width:
            raise ValueError(f'Feature id must be in [0, {self.settings.sae_width - 1}].')

        self.ensure_ready(preload_saes=False)
        assert self.model is not None and self.tokenizer is not None and self.sae_store is not None
        conditions: list[tuple[str, str, str]] = []
        for stem in stem_list:
            for cue in cue_list:
                conditions.append((stem, cue, self._cue_prompt(stem, cue)))
        prompts = [full for _, _, full in conditions]
        batch = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.settings.max_prompt_tokens,
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        capture: dict = {}
        with self._capture_hook(int(layer), capture):
            self.model(**batch, use_cache=False)
        sae = self.sae_store.get(int(layer))
        encoding = sae.encode(capture['hidden'])
        attention = batch.get('attention_mask', torch.ones_like(batch['input_ids'])).bool()

        rows: list[list[object]] = []
        chart_rows: list[list[object]] = []
        active_count = 0
        cue_active_context_counts = {cue: 0 for cue in cue_list}
        cue_activation_values = {cue: [] for cue in cue_list}
        for row_idx, (stem, cue, full_prompt) in enumerate(conditions):
            valid_positions = torch.nonzero(attention[row_idx], as_tuple=False).flatten()
            final_pos = int(valid_positions[-1].item())
            indices = encoding.indices[row_idx, final_pos]
            values = encoding.values[row_idx, final_pos]
            mask = indices == int(feature_id)
            activation = float(values[mask][0].item()) if bool(mask.any()) else 0.0
            active = activation > 0
            active_count += int(active)
            cue_active_context_counts[cue] += int(active)
            cue_activation_values[cue].append(activation)
            token_id = int(batch['input_ids'][row_idx, final_pos].item())
            final_token = self.tokenizer.decode([token_id])
            short_stem = stem if len(stem) <= 42 else stem[:39] + '…'
            rows.append([stem, cue, full_prompt, repr(final_token), activation, active])
            chart_rows.append([short_stem, cue, activation])

        cue_mean_activations = {
            cue: float(sum(vals) / len(vals)) if vals else 0.0
            for cue, vals in cue_activation_values.items()
        }
        dominant_cue = max(
            cue_list,
            key=lambda cue: (cue_active_context_counts[cue], cue_mean_activations[cue]),
        ) if cue_list else None
        dominant_count = cue_active_context_counts.get(dominant_cue, 0) if dominant_cue else 0
        off_dominant_active_count = sum(
            count for cue, count in cue_active_context_counts.items() if cue != dominant_cue
        )
        return FeatureCueContextResult(
            feature_id=int(feature_id),
            layer=int(layer),
            stems=stem_list,
            cues=cue_list,
            rows=rows,
            chart_rows=chart_rows,
            active_condition_count=active_count,
            condition_count=len(conditions),
            cue_active_context_counts=cue_active_context_counts,
            cue_mean_activations=cue_mean_activations,
            dominant_cue=dominant_cue,
            dominant_cue_context_count=dominant_count,
            off_dominant_active_count=off_dominant_active_count,
        )


RUNTIME = FeatureLensRuntime()

if SETTINGS.eager_load:
    try:
        RUNTIME.ensure_ready(preload_saes=True)
    except Exception as exc:  # Keep the UI alive and surface a useful error on first call.
        RUNTIME.load_error = f'{type(exc).__name__}: {exc}'
        print(f'FeatureLens eager load failed: {RUNTIME.load_error}')
