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

