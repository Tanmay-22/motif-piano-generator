from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
import torch.nn.functional as F
from torch import nn

from .config import V2ModelConfig
from .tokenizer import EventType


FEATURE_NAMES = (
    "event_type",
    "delta_coarse",
    "delta_fine",
    "pitch",
    "duration_coarse",
    "duration_fine",
    "velocity",
)


@dataclass(frozen=True)
class FactorizedLogits:
    event_type: torch.Tensor
    delta_coarse: torch.Tensor
    delta_fine: torch.Tensor
    pitch: torch.Tensor
    duration_coarse: torch.Tensor
    duration_fine: torch.Tensor
    velocity: torch.Tensor

    def feature(self, name: str) -> torch.Tensor:
        return getattr(self, name)


@dataclass(frozen=True)
class MotifMemory:
    hidden: torch.Tensor
    padding_mask: torch.Tensor


@dataclass(frozen=True)
class FactorizedLoss:
    total: torch.Tensor
    event_type: torch.Tensor
    delta_coarse: torch.Tensor
    delta_fine: torch.Tensor
    pitch: torch.Tensor
    duration_coarse: torch.Tensor
    duration_fine: torch.Tensor
    velocity: torch.Tensor
    valid_events: int
    note_events: int


class CompoundEventEmbedding(nn.Module):
    def __init__(self, config: V2ModelConfig) -> None:
        super().__init__()
        self.embeddings = nn.ModuleDict(
            {
                "event_type": nn.Embedding(config.event_type_vocab_size, config.model_dim, padding_idx=0),
                "delta_coarse": nn.Embedding(config.delta_coarse_vocab_size, config.model_dim, padding_idx=0),
                "delta_fine": nn.Embedding(config.delta_fine_vocab_size, config.model_dim, padding_idx=0),
                "pitch": nn.Embedding(config.pitch_vocab_size, config.model_dim, padding_idx=0),
                "duration_coarse": nn.Embedding(
                    config.duration_coarse_vocab_size, config.model_dim, padding_idx=0
                ),
                "duration_fine": nn.Embedding(
                    config.duration_fine_vocab_size, config.model_dim, padding_idx=0
                ),
                "velocity": nn.Embedding(config.velocity_vocab_size, config.model_dim, padding_idx=0),
            }
        )

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        if events.ndim != 3 or events.size(-1) != len(FEATURE_NAMES):
            raise ValueError("Compound events must have shape [batch, sequence, 7].")
        embedded = sum(
            self.embeddings[name](events[..., index])
            for index, name in enumerate(FEATURE_NAMES)
        )
        return embedded / sqrt(len(FEATURE_NAMES))


class MotifContinuationTransformer(nn.Module):
    """Compact seq2seq model with persistent motif cross-attention."""

    def __init__(self, config: V2ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.event_embedding = CompoundEventEmbedding(config)
        self.motif_positions = nn.Embedding(config.max_motif_events + 1, config.model_dim)
        self.continuation_positions = nn.Embedding(config.max_continuation_events, config.model_dim)
        self.category_embedding = nn.Embedding(config.category_vocab_size, config.model_dim)
        self.texture_embedding = nn.Embedding(config.texture_vocab_size, config.model_dim)
        self.control_projection = nn.Sequential(
            nn.Linear(config.control_feature_dim, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.model_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.model_dim,
            nhead=config.heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.encoder_layers,
            norm=nn.LayerNorm(config.model_dim),
            enable_nested_tensor=False,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.decoder_layers,
            norm=nn.LayerNorm(config.model_dim),
        )
        self.output_heads = nn.ModuleDict(
            {
                "event_type": nn.Linear(config.model_dim, config.event_type_vocab_size),
                "delta_coarse": nn.Linear(config.model_dim, config.delta_coarse_vocab_size),
                "delta_fine": nn.Linear(config.model_dim, config.delta_fine_vocab_size),
                "pitch": nn.Linear(config.model_dim, config.pitch_vocab_size),
                "duration_coarse": nn.Linear(config.model_dim, config.duration_coarse_vocab_size),
                "duration_fine": nn.Linear(config.model_dim, config.duration_fine_vocab_size),
                "velocity": nn.Linear(config.model_dim, config.velocity_vocab_size),
            }
        )
        self.apply(self._initialize_module)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _infer_padding(events: torch.Tensor) -> torch.Tensor:
        return events[..., 0].eq(int(EventType.PAD))

    def encode_motif(
        self,
        motif_events: torch.Tensor,
        category_ids: torch.Tensor,
        texture_ids: torch.Tensor,
        motif_controls: torch.Tensor,
        motif_padding_mask: torch.Tensor | None = None,
    ) -> MotifMemory:
        batch, sequence_length, _ = motif_events.shape
        if sequence_length > self.config.max_motif_events:
            raise ValueError("Motif sequence exceeds the configured maximum.")
        if motif_padding_mask is None:
            motif_padding_mask = self._infer_padding(motif_events)
        if motif_controls.shape != (batch, self.config.control_feature_dim):
            raise ValueError("Motif controls have the wrong shape.")

        control = (
            self.category_embedding(category_ids)
            + self.texture_embedding(texture_ids)
            + self.control_projection(motif_controls)
        ).unsqueeze(1)
        event_hidden = self.event_embedding(motif_events)
        hidden = torch.cat([control, event_hidden], dim=1)
        positions = torch.arange(sequence_length + 1, device=motif_events.device).unsqueeze(0)
        hidden = hidden + self.motif_positions(positions)
        control_mask = torch.zeros(batch, 1, dtype=torch.bool, device=motif_events.device)
        memory_padding = torch.cat([control_mask, motif_padding_mask], dim=1)
        memory = self.encoder(hidden, src_key_padding_mask=memory_padding)
        return MotifMemory(memory, memory_padding)

    def decode(
        self,
        continuation_events: torch.Tensor,
        memory: MotifMemory,
        continuation_padding_mask: torch.Tensor | None = None,
    ) -> FactorizedLogits:
        _, sequence_length, _ = continuation_events.shape
        if sequence_length > self.config.max_continuation_events:
            raise ValueError("Continuation sequence exceeds the configured maximum.")
        if continuation_padding_mask is None:
            continuation_padding_mask = self._infer_padding(continuation_events)
        positions = torch.arange(sequence_length, device=continuation_events.device).unsqueeze(0)
        hidden = self.event_embedding(continuation_events) + self.continuation_positions(positions)
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=hidden.device),
            diagonal=1,
        )
        decoded = self.decoder(
            hidden,
            memory.hidden,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=continuation_padding_mask,
            memory_key_padding_mask=memory.padding_mask,
            tgt_is_causal=True,
        )
        return FactorizedLogits(**{name: head(decoded) for name, head in self.output_heads.items()})

    def forward(
        self,
        motif_events: torch.Tensor,
        continuation_events: torch.Tensor,
        category_ids: torch.Tensor,
        texture_ids: torch.Tensor,
        motif_controls: torch.Tensor,
        motif_padding_mask: torch.Tensor | None = None,
        continuation_padding_mask: torch.Tensor | None = None,
    ) -> FactorizedLogits:
        memory = self.encode_motif(
            motif_events,
            category_ids,
            texture_ids,
            motif_controls,
            motif_padding_mask,
        )
        return self.decode(continuation_events, memory, continuation_padding_mask)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def factorized_event_loss(logits: FactorizedLogits, targets: torch.Tensor) -> FactorizedLoss:
    if targets.ndim != 3 or targets.size(-1) != len(FEATURE_NAMES):
        raise ValueError("Targets must have shape [batch, sequence, 7].")
    valid_mask = targets[..., 0].ne(int(EventType.PAD))
    note_mask = targets[..., 0].eq(int(EventType.NOTE)) & valid_mask

    def masked_loss(values: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not bool(mask.any()):
            return values.sum() * 0.0
        per_event = F.cross_entropy(
            values.reshape(-1, values.size(-1)),
            labels.reshape(-1),
            reduction="none",
        ).view_as(labels)
        return per_event[mask].mean()

    event_loss = masked_loss(logits.event_type, targets[..., 0], valid_mask)
    note_losses = {
        name: masked_loss(logits.feature(name), targets[..., index], note_mask)
        for index, name in enumerate(FEATURE_NAMES[1:], start=1)
    }
    note_average = torch.stack(tuple(note_losses.values())).mean()
    total = event_loss + note_average
    return FactorizedLoss(
        total=total,
        event_type=event_loss,
        **note_losses,
        valid_events=int(valid_mask.sum().item()),
        note_events=int(note_mask.sum().item()),
    )
