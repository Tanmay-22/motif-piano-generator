from __future__ import annotations

import torch
from torch import nn
from types import SimpleNamespace

from motifgen.config import DataConfig, GenerationConfig, ModelConfig
from motifgen.generation import MotifGenerator


class PredictTimeShiftModel(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.config = ModelConfig(vocab_size=tokenizer.vocab_size, embed_dim=16, heads=2, layers=1)
        self.tokenizer = tokenizer

    def forward(self, tokens, padding_mask=None):
        batch, length = tokens.shape
        logits = torch.full((batch, length, self.tokenizer.vocab_size), -20.0, device=tokens.device)
        logits[:, :, self.tokenizer.pad_id] = 50.0
        logits[:, :, self.tokenizer.token_to_id["TIME_SHIFT_100"]] = 40.0
        return logits

    def prefill(self, tokens, padding_mask=None):
        return self.forward(tokens, padding_mask), SimpleNamespace(sequence_length=tokens.size(1))

    def forward_step(self, token, state, is_padding=None):
        return self.forward(token, is_padding), SimpleNamespace(sequence_length=state.sequence_length + 1)


def test_generation_masks_forbidden_tokens_and_reaches_duration(tokenizer):
    model = PredictTimeShiftModel(tokenizer)
    generator = MotifGenerator(
        model,
        tokenizer,
        DataConfig(motif_min_tokens=2, motif_max_tokens=4),
        GenerationConfig(max_generated_tokens=20, top_k=8),
    )
    motif = [tokenizer.token_to_id["VEL_20"], tokenizer.token_to_id["NOTE_ON_60"]]
    result = generator.generate(motif, target_seconds=5, temperature=1.0, seed=7)
    assert result.reached_target_duration
    assert result.duration_seconds >= 5
    assert not tokenizer.forbidden_generation_ids.intersection(result.continuation_tokens)
