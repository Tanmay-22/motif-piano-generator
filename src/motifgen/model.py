from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig


@dataclass
class IncrementalState:
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    padding_mask: torch.Tensor

    @property
    def sequence_length(self) -> int:
        return self.padding_mask.size(1)


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

    @staticmethod
    def _project_attention(
        hidden: torch.Tensor, attention: nn.MultiheadAttention
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence_length, embed_dim = hidden.shape
        projected = F.linear(hidden, attention.in_proj_weight, attention.in_proj_bias)
        query, key, value = projected.chunk(3, dim=-1)
        heads = attention.num_heads
        head_dim = embed_dim // heads

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, sequence_length, heads, head_dim).transpose(1, 2)

        return split_heads(query), split_heads(key), split_heads(value)

    def prefill(
        self, tokens: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, IncrementalState]:
        """Run a prompt once and retain per-layer keys/values for fast decoding."""
        batch, sequence_length = tokens.shape
        if sequence_length > self.config.max_sequence_length:
            raise ValueError(f"Sequence length {sequence_length} exceeds model maximum {self.config.max_sequence_length}.")
        if padding_mask is None:
            padding_mask = torch.zeros_like(tokens, dtype=torch.bool)

        positions = torch.arange(sequence_length, device=tokens.device).unsqueeze(0)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        cached_keys: list[torch.Tensor] = []
        cached_values: list[torch.Tensor] = []
        for layer in self.transformer.layers:
            normalized = layer.norm1(hidden)
            _, key, value = self._project_attention(normalized, layer.self_attn)
            cached_keys.append(key)
            cached_values.append(value)
            hidden = layer(
                hidden,
                src_mask=causal_mask,
                src_key_padding_mask=padding_mask,
                is_causal=True,
            )
        logits = self.output(self.normalization(hidden))
        return logits, IncrementalState(cached_keys, cached_values, padding_mask)

    def forward_step(
        self,
        token: torch.Tensor,
        state: IncrementalState,
        is_padding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, IncrementalState]:
        """Decode one token using cached attention without recomputing prior tokens."""
        if token.shape[1] != 1:
            raise ValueError("forward_step expects exactly one token per batch item.")
        if state.sequence_length >= self.config.max_sequence_length:
            raise ValueError("Incremental cache is full; prefill a shorter context before continuing.")
        if is_padding is None:
            is_padding = torch.zeros(token.size(0), 1, dtype=torch.bool, device=token.device)

        position = torch.tensor([[state.sequence_length]], device=token.device)
        hidden = self.token_embedding(token) + self.position_embedding(position)
        padding_mask = torch.cat([state.padding_mask, is_padding], dim=1)
        next_keys: list[torch.Tensor] = []
        next_values: list[torch.Tensor] = []

        for layer_index, layer in enumerate(self.transformer.layers):
            normalized = layer.norm1(hidden)
            query, key, value = self._project_attention(normalized, layer.self_attn)
            keys = torch.cat([state.keys[layer_index], key], dim=2)
            values = torch.cat([state.values[layer_index], value], dim=2)
            scores = torch.matmul(query, keys.transpose(-2, -1)) / sqrt(query.size(-1))
            scores = scores.masked_fill(padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
            attention_weights = torch.softmax(scores, dim=-1)
            attended = torch.matmul(attention_weights, values)
            attended = attended.transpose(1, 2).contiguous().view(token.size(0), 1, self.config.embed_dim)
            attended = F.linear(attended, layer.self_attn.out_proj.weight, layer.self_attn.out_proj.bias)
            hidden = hidden + layer.dropout1(attended)
            feedforward = layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(hidden)))))
            hidden = hidden + layer.dropout2(feedforward)
            next_keys.append(keys)
            next_values.append(values)

        logits = self.output(self.normalization(hidden))
        return logits, IncrementalState(next_keys, next_values, padding_mask)
