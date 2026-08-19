from __future__ import annotations

import os
from dataclasses import dataclass


def _parse_layers(raw: str | None) -> tuple[int, ...]:
    if not raw:
        return (4, 14, 26)
    layers = tuple(sorted({int(x.strip()) for x in raw.split(',') if x.strip()}))
    if not layers:
        raise ValueError('FEATURELENS_LAYERS must contain at least one layer.')
    if any(layer < 0 or layer > 27 for layer in layers):
        raise ValueError('Qwen3-1.7B has residual-stream SAE layers 0-27.')
    return layers


@dataclass(frozen=True)
class Settings:
    model_id: str = os.getenv('FEATURELENS_MODEL_ID', 'Qwen/Qwen3-1.7B-Base')
    sae_repo_id: str = os.getenv(
        'FEATURELENS_SAE_REPO', 'Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50'
    )
    layers: tuple[int, ...] = _parse_layers(os.getenv('FEATURELENS_LAYERS'))
    sae_top_k: int = int(os.getenv('FEATURELENS_SAE_TOP_K', '50'))
    sae_width: int = 32_768
    d_model: int = 2_048
    max_prompt_tokens: int = int(os.getenv('FEATURELENS_MAX_PROMPT_TOKENS', '256'))
    max_new_tokens: int = int(os.getenv('FEATURELENS_MAX_NEW_TOKENS', '32'))
    live_random_controls: int = int(os.getenv('FEATURELENS_LIVE_RANDOM_CONTROLS', '8'))
    contrast_prompts_per_concept: int = int(os.getenv('FEATURELENS_CONTRAST_PROMPTS_PER_CONCEPT', '4'))
    eager_load: bool = os.getenv(
        'FEATURELENS_EAGER_LOAD', '1' if os.getenv('SPACE_ID') else '0'
    ).lower() in {'1', 'true', 'yes', 'on'}
    sae_dtype: str = os.getenv(
        'FEATURELENS_SAE_DTYPE', 'float16' if os.getenv('SPACE_ID') else 'float32'
    )


SETTINGS = Settings()
