"""Version 2 musical representation and motif-conditioned model components."""

from .config import V2GenerationConfig, V2ModelConfig
from .controls import category_id, motif_feature_vector, texture_id
from .features import MotifFeatures, TextureClass, extract_motif_features
from .generation import V2GenerationResult, V2MotifGenerator
from .model import (
    FactorizedLogits,
    FactorizedLoss,
    MotifContinuationTransformer,
    MotifMemory,
    factorized_event_loss,
)
from .tokenizer import (
    CompleteNoteEvent,
    CompleteNoteTokenizer,
    EventType,
    MusicCategory,
)

__all__ = [
    "CompleteNoteEvent",
    "CompleteNoteTokenizer",
    "EventType",
    "FactorizedLogits",
    "FactorizedLoss",
    "MotifFeatures",
    "MotifContinuationTransformer",
    "MotifMemory",
    "MusicCategory",
    "TextureClass",
    "V2GenerationConfig",
    "V2GenerationResult",
    "V2ModelConfig",
    "V2MotifGenerator",
    "category_id",
    "extract_motif_features",
    "factorized_event_loss",
    "motif_feature_vector",
    "texture_id",
]
