from __future__ import annotations

import torch

from motifgen.config import ModelConfig
from motifgen.model import MusicTransformer


def test_transformer_forward_shape():
    config = ModelConfig(vocab_size=32, embed_dim=16, heads=2, layers=1, feedforward_dim=32, max_sequence_length=16)
    model = MusicTransformer(config)
    inputs = torch.randint(0, config.vocab_size, (2, 8))
    padding = torch.zeros_like(inputs, dtype=torch.bool)
    assert model(inputs, padding).shape == (2, 8, config.vocab_size)


def test_incremental_decoding_matches_full_forward():
    torch.manual_seed(9)
    config = ModelConfig(
        vocab_size=32,
        embed_dim=16,
        heads=2,
        layers=2,
        feedforward_dim=32,
        max_sequence_length=16,
        dropout=0,
    )
    model = MusicTransformer(config).eval()
    prompt = torch.tensor([[1, 8, 9, 3]])
    padding = torch.zeros_like(prompt, dtype=torch.bool)
    full_prompt = model(prompt, padding)
    cached_prompt, state = model.prefill(prompt, padding)
    assert torch.allclose(cached_prompt, full_prompt, atol=1e-5)

    next_token = torch.tensor([[12]])
    incremental, _ = model.forward_step(next_token, state)
    complete = torch.cat([prompt, next_token], dim=1)
    complete_padding = torch.zeros_like(complete, dtype=torch.bool)
    expected = model(complete, complete_padding)[:, -1:]
    assert torch.allclose(incremental, expected, atol=1e-5)
