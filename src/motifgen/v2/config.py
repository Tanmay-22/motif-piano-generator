from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .controls import CATEGORY_ORDER, CONTROL_FEATURE_NAMES, TEXTURE_ORDER
from .tokenizer import CompleteNoteTokenizer, EventType


@dataclass(frozen=True)
class V2ModelConfig:
    event_type_vocab_size: int = len(EventType)
    delta_coarse_vocab_size: int = 32
    delta_fine_vocab_size: int = 101
    pitch_vocab_size: int = 89
    duration_coarse_vocab_size: int = 32
    duration_fine_vocab_size: int = 101
    velocity_vocab_size: int = 33
    category_vocab_size: int = len(CATEGORY_ORDER)
    texture_vocab_size: int = len(TEXTURE_ORDER)
    control_feature_dim: int = len(CONTROL_FEATURE_NAMES)
    model_dim: int = 192
    heads: int = 6
    encoder_layers: int = 3
    decoder_layers: int = 4
    feedforward_dim: int = 768
    dropout: float = 0.1
    max_motif_events: int = 130
    max_continuation_events: int = 258

    def __post_init__(self) -> None:
        if self.model_dim % self.heads:
            raise ValueError("model_dim must be divisible by heads.")
        if min(self.encoder_layers, self.decoder_layers, self.max_motif_events) < 1:
            raise ValueError("Layer and sequence counts must be positive.")
        if self.max_continuation_events < 2:
            raise ValueError("max_continuation_events must include at least BOS and EOS.")

    @classmethod
    def from_tokenizer(cls, tokenizer: CompleteNoteTokenizer, **overrides: Any) -> "V2ModelConfig":
        sizes = tokenizer.feature_sizes
        values = {
            "event_type_vocab_size": sizes["event_type"],
            "delta_coarse_vocab_size": sizes["delta_coarse"],
            "delta_fine_vocab_size": sizes["delta_fine"],
            "pitch_vocab_size": sizes["pitch"],
            "duration_coarse_vocab_size": sizes["duration_coarse"],
            "duration_fine_vocab_size": sizes["duration_fine"],
            "velocity_vocab_size": sizes["velocity"],
            **overrides,
        }
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "V2ModelConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})


@dataclass(frozen=True)
class V2GenerationConfig:
    allowed_durations: tuple[int, ...] = (5, 10, 20)
    min_temperature: float = 0.6
    max_temperature: float = 1.4
    top_k: int = 16
    top_p: float = 0.9
    wall_clock_seconds: float = 35.0

    def __post_init__(self) -> None:
        if not self.allowed_durations or min(self.allowed_durations) < 1:
            raise ValueError("Generation durations must be positive.")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be between zero and one.")
        if self.top_k < 1 or self.wall_clock_seconds <= 0:
            raise ValueError("Generation limits must be positive.")
