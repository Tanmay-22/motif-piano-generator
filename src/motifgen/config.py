from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 313
    embed_dim: int = 256
    heads: int = 4
    layers: int = 4
    feedforward_dim: int = 512
    max_sequence_length: int = 512
    dropout: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})


@dataclass(frozen=True)
class DataConfig:
    motif_min_tokens: int = 16
    motif_max_tokens: int = 64
    continuation_tokens: int = 256
    sample_rate: int = 100
    max_time_shift_steps: int = 100

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "DataConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})


@dataclass(frozen=True)
class GenerationConfig:
    allowed_durations: tuple[int, ...] = (5, 10, 20)
    min_temperature: float = 0.6
    max_temperature: float = 1.4
    max_generated_tokens: int = 1400
    top_k: int = 32
    cache_reset_tokens: int = 256
