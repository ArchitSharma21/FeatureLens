from __future__ import annotations

import math
import sys
import types
from types import SimpleNamespace

import torch

try:
    import transformers  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType('transformers')
    stub.AutoModelForCausalLM = type('AutoModelForCausalLM', (), {})
    stub.AutoTokenizer = type('AutoTokenizer', (), {})
    sys.modules['transformers'] = stub

from featurelens.config import Settings
from featurelens.runtime import FeatureLensRuntime
from featurelens.sae import SAEWeights


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    eos_token = '<pad>'
    pad_token = '<pad>'
    padding_side = 'left'

    @staticmethod
    def _ids(text: str) -> list[int]:
        # Keep 0 reserved for padding; make deterministic small-vocabulary ids.
        return [1 + (ord(char) % 7) for char in text] or [1]

    def __call__(
        self,
        text,
        *,
        return_tensors=None,
        padding=False,
        truncation=False,
        max_length=None,
        add_special_tokens=True,
    ):
        if isinstance(text, str):
            ids = self._ids(text)
            if max_length is not None:
                ids = ids[-int(max_length) :]
            if return_tensors == 'pt':
                tensor = torch.tensor([ids], dtype=torch.long)
                return {'input_ids': tensor, 'attention_mask': torch.ones_like(tensor)}
            return {'input_ids': ids}

        sequences = [self._ids(item) for item in text]
        if max_length is not None:
            sequences = [ids[-int(max_length) :] for ids in sequences]
        width = max(len(ids) for ids in sequences)
        padded = []
        masks = []
        for ids in sequences:
            pad = width - len(ids)
            padded.append([0] * pad + ids)
            masks.append([0] * pad + [1] * len(ids))
        return {
            'input_ids': torch.tensor(padded, dtype=torch.long),
            'attention_mask': torch.tensor(masks, dtype=torch.long),
        }

    def decode(self, ids) -> str:
        token_id = int(ids[0])
        return '<pad>' if token_id == 0 else f't{token_id}'


class FakeBackbone(torch.nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Identity()])
        self.embedding = torch.nn.Embedding(8, d_model)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.2, 0.1],
                        [0.8, 0.3, 0.2],
                        [0.6, 0.4, 0.3],
                        [0.4, 0.5, 0.4],
                        [0.3, 0.6, 0.5],
                        [0.2, 0.7, 0.6],
                        [0.1, 0.8, 0.7],
                    ]
                )
            )


class FakeLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeBackbone(d_model=3)
        self.proj = torch.nn.Linear(3, 8, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.5, 0.5, 0.0],
                        [0.5, 0.0, 0.5],
                        [0.0, 0.5, 0.5],
                        [-0.4, 0.3, 0.2],
                    ]
                )
            )

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden = self.model.embedding(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=self.proj(hidden))


class FakeSAEStore:
    def __init__(self) -> None:
        self.sae = SAEWeights(
            layer=0,
            w_enc_t=torch.tensor(
                [
                    [1.0, 0.0, 0.3, -0.2],
                    [0.0, 1.0, 0.2, 0.1],
                    [0.0, 0.0, 1.0, 0.4],
                ]
            ),
            w_dec=torch.tensor(
                [
                    [1.0, 0.0, 0.5, -0.2],
                    [0.0, 1.0, 0.2, 0.3],
                    [0.0, 0.0, 1.0, 0.4],
                ]
            ),
            b_enc=torch.zeros(4),
            b_dec=torch.zeros(3),
            top_k=2,
        )

    def get(self, layer: int):
        assert layer == 0
        return self.sae

    def preload(self) -> None:
        return None


def make_runtime() -> FeatureLensRuntime:
    settings = Settings(
        layers=(0,),
        sae_top_k=2,
        sae_width=4,
        d_model=3,
        max_prompt_tokens=32,
        live_random_controls=3,
        contrast_prompts_per_concept=2,
        eager_load=False,
        sae_dtype='float32',
    )
    runtime = FeatureLensRuntime(settings)
    runtime.device = torch.device('cpu')
    runtime.model = FakeLM().eval()
    runtime.tokenizer = FakeTokenizer()
    runtime.sae_store = FakeSAEStore()
    return runtime


def test_feature_token_trace_runs_end_to_end_on_toy_runtime() -> None:
    runtime = make_runtime()
    result = runtime.feature_token_trace('abc', layer=0, feature_id=0)
    assert result.token_count == 3
    assert len(result.rows) == 3
    assert result.active_token_count >= 1
    assert result.max_activation > 0


def test_feature_geometry_runs_end_to_end_on_toy_runtime() -> None:
    runtime = make_runtime()
    result = runtime.feature_geometry('abc', layer=0, token_index=-1, feature_ids=[0, 1, 2])
    assert len(result.rows) == 3  # 3 choose 2
    assert math.isfinite(result.alignment_ratio)
    assert result.independent_norm >= 0


def test_contrastive_intervention_runs_end_to_end_on_toy_runtime() -> None:
    runtime = make_runtime()
    result = runtime.contrastive_intervention(
        text='abc',
        layer=0,
        token_index=-1,
        feature_id=0,
        mode='ablate',
        coefficient=0.0,
        target_a='a',
        target_b='b',
    )
    assert len(result.rows) == 2
    assert result.random_control_count == 3
    assert math.isfinite(result.delta_log_odds)
    assert math.isfinite(result.specificity_ratio)


def test_concept_contrast_promptwide_scan_runs_on_toy_runtime() -> None:
    runtime = make_runtime()
    result = runtime.concept_contrast_scan(feature_id=0, layer=0, prompts_per_concept=1)
    assert result.total_prompt_count == 7
    assert len(result.rows) == 7
    assert all(len(row) == 7 for row in result.rows)
    assert 0 <= result.active_prompt_count <= result.total_prompt_count


def test_concept_feature_discovery_runs_on_toy_runtime() -> None:
    runtime = make_runtime()
    result = runtime.concept_feature_discovery(
        concept='mathematics',
        layer=0,
        prompts_per_concept=1,
        top_n=3,
        ranking_mode='balanced_selectivity',
        current_text='abc',
        current_token_index=-1,
    )
    assert result.concept == 'mathematics'
    assert result.ranking_mode == 'balanced_selectivity'
    assert result.current_context_available is True
    assert result.current_token_index == 2
    assert len(result.rows) <= 3
    assert result.candidate_ids == [int(row[1]) for row in result.rows]
    assert all(len(row) == 14 for row in result.rows)
    if result.rows:
        assert result.default_candidate_id in result.candidate_ids
        assert all(math.isfinite(float(row[2])) for row in result.rows)
        assert all(float(row[9]) >= 0 for row in result.rows)  # current prompt max
        assert all(float(row[10]) >= 0 for row in result.rows)  # current token activation


def test_concept_feature_discovery_supports_raw_mean_difference() -> None:
    runtime = make_runtime()
    result = runtime.concept_feature_discovery(
        concept='mathematics',
        layer=0,
        prompts_per_concept=1,
        top_n=3,
        ranking_mode='raw_mean_difference',
    )
    assert result.ranking_mode == 'raw_mean_difference'
    assert result.current_context_available is False
    assert result.current_token_index is None
    assert all(len(row) == 14 for row in result.rows)




def test_concept_feature_discovery_supports_causal_ready_mode() -> None:
    runtime = make_runtime()
    result = runtime.concept_feature_discovery(
        concept='mathematics',
        layer=0,
        prompts_per_concept=1,
        top_n=5,
        ranking_mode='causal_ready',
        current_text='abc',
        current_token_index=-1,
    )
    assert result.ranking_mode == 'causal_ready'
    assert result.current_context_available is True
    assert result.displayed_current_active_count == len(result.rows)
    assert all(bool(row[11]) for row in result.rows)


def test_concept_feature_discovery_causal_ready_requires_workbench_context() -> None:
    runtime = make_runtime()
    try:
        runtime.concept_feature_discovery(
            concept='mathematics',
            layer=0,
            prompts_per_concept=1,
            top_n=3,
            ranking_mode='causal_ready',
        )
    except ValueError as exc:
        assert 'Workbench' in str(exc)
    else:
        raise AssertionError('causal_ready should require Workbench context')


def test_feature_cue_scan_runs_on_toy_runtime() -> None:
    runtime = make_runtime()
    result = runtime.feature_cue_scan(
        feature_id=0,
        layer=0,
        prompt_stem='abc',
        cues=['is', '=', ':'],
    )
    assert result.cue_count == 3
    assert len(result.rows) == 3
    assert all(len(row) == 5 for row in result.rows)
    assert 0 <= result.active_cue_count <= result.cue_count


def test_feature_cue_context_scan_runs_on_toy_runtime() -> None:
    runtime = make_runtime()
    result = runtime.feature_cue_context_scan(
        feature_id=0,
        layer=0,
        stems=['abc', 'xyz'],
        cues=['is', '=', ':'],
    )
    assert result.condition_count == 6
    assert len(result.rows) == 6
    assert all(len(row) == 6 for row in result.rows)
    assert 0 <= result.active_condition_count <= result.condition_count
    assert set(result.cue_active_context_counts) == {'is', '=', ':'}
    assert set(result.cue_mean_activations) == {'is', '=', ':'}
    assert result.dominant_cue in {'is', '=', ':'}
    assert 0 <= result.dominant_cue_context_count <= 2
    assert 0 <= result.off_dominant_active_count <= result.active_condition_count
    assert len(result.chart_rows) == 6


def test_candidate_causal_screen_batches_multiple_ablation_candidates() -> None:
    runtime = make_runtime()
    result = runtime.candidate_causal_screen(
        text='abc',
        layer=0,
        token_index=-1,
        feature_ids=[0, 1, 2],
        target_text='d',
    )
    assert result.candidate_count == 3
    assert len(result.rows) == 3
    assert len(result.chart_rows) == 3
    assert all(len(row) == 8 for row in result.rows)
    assert all(row[0] == rank for rank, row in enumerate(result.rows, start=1))
    assert 0 <= result.active_feature_count <= result.candidate_count
    assert all(math.isfinite(float(row[5])) for row in result.rows)
    assert all(float(row[7]) >= 0 for row in result.rows)


def test_candidate_specificity_screen_batches_random_controlled_candidates() -> None:
    runtime = make_runtime()
    result = runtime.candidate_specificity_screen(
        text='abc',
        layer=0,
        token_index=-1,
        feature_ids=[0, 1],
        target_text='d',
    )
    assert result.candidate_count == 2
    assert result.random_control_count == 3
    assert len(result.rows) == 2
    assert len(result.chart_rows) == 4
    assert all(len(row) == 17 for row in result.rows)
    assert all(row[0] == rank for rank, row in enumerate(result.rows, start=1))
    assert all(float(row[9]) >= 0 for row in result.rows)  # target specificity
    assert all(0 < float(row[10]) <= 1 for row in result.rows)  # empirical tail
    assert all(float(row[15]) >= 0 for row in result.rows)  # JS specificity
    assert all(0 < float(row[16]) <= 1 for row in result.rows)


def test_concept_feature_discovery_reports_split_half_stability_without_extra_forward() -> None:
    runtime = make_runtime()
    result = runtime.concept_feature_discovery(
        concept='mathematics',
        layer=0,
        prompts_per_concept=2,
        top_n=3,
        ranking_mode='balanced_selectivity',
        current_text='abc',
        current_token_index=-1,
    )
    if result.split_half_jaccard is not None:
        assert 0.0 <= result.split_half_jaccard <= 1.0
        assert result.split_half_k is not None
        assert 0 <= result.split_half_shared_count <= max(len(result.candidate_ids), result.split_half_k)


def test_concept_feature_discovery_reports_resample_support_from_same_batch() -> None:
    runtime = make_runtime()
    result = runtime.concept_feature_discovery(
        concept='mathematics',
        layer=0,
        prompts_per_concept=2,
        top_n=3,
        ranking_mode='balanced_selectivity',
        current_text='abc',
        current_token_index=-1,
    )
    assert 0 <= result.resample_replicates <= 32
    if result.resample_replicates and result.rows:
        assert result.resample_mean_support is not None
        assert 0.0 <= result.resample_mean_support <= 1.0
        assert 0 <= result.resample_high_support_count <= len(result.rows)
        for row in result.rows:
            assert row[12] is not None
            assert 0.0 <= float(row[12]) <= 1.0
            if row[13] is not None:
                assert 1.0 <= float(row[13]) <= result.top_n


def test_candidate_cross_target_profile_runs_multiple_features_and_targets() -> None:
    runtime = make_runtime()
    result = runtime.candidate_cross_target_profile(
        text='abc',
        layer=0,
        token_index=-1,
        feature_ids=[0, 1],
        targets=['d', 'e', 'f'],
    )
    assert result.feature_ids == [0, 1]
    assert result.targets == ['d', 'e', 'f']
    assert len(result.rows) == 6
    assert len(result.chart_rows) == 6
    assert len(result.summary_rows) == 2
    assert all(len(row) == 8 for row in result.rows)
    assert all(len(row) == 12 for row in result.summary_rows)
    assert len(result.pairwise_rows) == 2 * 3  # 2 features × C(3 targets, 2)
    assert all(len(row) == 6 for row in result.pairwise_rows)
    assert all(0.0 <= float(row[7]) <= 1.0 for row in result.summary_rows)  # normalized entropy
    assert all(0.0 <= float(row[8]) <= 1.0 for row in result.summary_rows)  # concentration
    assert all(-1.0 <= float(row[9]) <= 1.0 for row in result.summary_rows)  # signed bias
    assert all(isinstance(row[10], str) and row[10] for row in result.summary_rows)
    assert all(math.isfinite(float(row[5])) for row in result.rows)
