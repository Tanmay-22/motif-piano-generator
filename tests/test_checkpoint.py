from __future__ import annotations

import torch

from motifgen.config import DataConfig, ModelConfig
from motifgen.generation import MotifGenerator
from motifgen.model import MusicTransformer


def test_checkpoint_loads_with_matching_configuration(tmp_path):
    model_config = ModelConfig(
        vocab_size=313,
        embed_dim=16,
        heads=2,
        layers=1,
        feedforward_dim=32,
        max_sequence_length=128,
        dropout=0,
    )
    data_config = DataConfig(motif_min_tokens=2, motif_max_tokens=4, continuation_tokens=8)
    model = MusicTransformer(model_config)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model_config.to_dict(),
            "data_config": data_config.to_dict(),
        },
        checkpoint,
    )
    generator = MotifGenerator.from_checkpoint(checkpoint)
    assert generator.model.config == model_config
    assert generator.data_config == data_config
