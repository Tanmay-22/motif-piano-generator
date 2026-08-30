from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig


class MusicTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.layers, enable_nested_tensor=False)
        self.normalization = nn.LayerNorm(config.embed_dim)
        self.output = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        _, sequence_length = tokens.shape
        if sequence_length > self.config.max_sequence_length:
            raise ValueError(f"Sequence length {sequence_length} exceeds model maximum {self.config.max_sequence_length}.")
        positions = torch.arange(sequence_length, device=tokens.device).unsqueeze(0)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        hidden = self.transformer(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)
        return self.output(self.normalization(hidden))

