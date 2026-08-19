from __future__ import annotations

import torch

from featurelens.sae import SAEWeights


def make_sae() -> SAEWeights:
    w_enc_t = torch.tensor(
        [
            [1.0, 0.0, -1.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 0.5, -1.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
        ]
    )
    w_dec = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
        ]
    )
    return SAEWeights(
        layer=0,
        w_enc_t=w_enc_t,
        w_dec=w_dec,
        b_enc=torch.zeros(5),
        b_dec=torch.zeros(3),
        top_k=2,
    )


def test_encode_keeps_topk_relu_features() -> None:
    sae = make_sae()
    encoding = sae.encode(torch.tensor([2.0, 1.0, -1.0]))
    assert encoding.indices.shape == (2,)
    assert set(encoding.indices.tolist()) == {0, 3}
    assert encoding.active_count == 2
    assert encoding.activation_for(0) == 2.0
    assert encoding.activation_for(4) == 0.0


def test_decode_sparse_uses_selected_decoder_columns() -> None:
    sae = make_sae()
    encoding = sae.encode(torch.tensor([2.0, 1.0, -1.0]))
    reconstructed = sae.decode_sparse(encoding)
    assert reconstructed.shape == (3,)
    assert torch.isfinite(reconstructed).all()


def test_batched_encode_and_decode() -> None:
    sae = make_sae()
    hidden = torch.tensor([[2.0, 1.0, -1.0], [0.0, 2.0, 2.0]])
    encoding = sae.encode(hidden)
    decoded = sae.decode_sparse(encoding)
    assert encoding.indices.shape == (2, 2)
    assert decoded.shape == hidden.shape
