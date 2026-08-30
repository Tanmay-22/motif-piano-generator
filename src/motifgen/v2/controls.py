from __future__ import annotations

from typing import Final

from .features import MotifFeatures, TextureClass
from .tokenizer import MusicCategory


CATEGORY_ORDER: Final = (
    MusicCategory.AUTO,
    MusicCategory.BAROQUE_CLASSICAL,
    MusicCategory.ROMANTIC,
    MusicCategory.IMPRESSIONIST_MODERN,
)
TEXTURE_ORDER: Final = (
    TextureClass.MONOPHONIC,
    TextureClass.LIGHT_POLYPHONIC,
    TextureClass.FULL_POLYPHONIC,
)
CONTROL_FEATURE_NAMES: Final = (
    "duration",
    "note_density",
    "onset_density",
    "average_polyphony",
    "peak_polyphony",
    "average_chord_size",
    "pitch_mean",
    "pitch_span",
    "velocity_mean",
    "velocity_range",
    "median_onset_gap",
    "median_note_duration",
    "bass_and_treble",
)


def category_id(category: MusicCategory) -> int:
    return CATEGORY_ORDER.index(category)


def texture_id(texture: TextureClass) -> int:
    return TEXTURE_ORDER.index(texture)


def _unit(value: float, maximum: float) -> float:
    return max(0.0, min(1.0, value / maximum))


def motif_feature_vector(features: MotifFeatures) -> tuple[float, ...]:
    """Normalize interpretable motif measurements to a stable [0, 1] vector."""

    vector = (
        _unit(features.duration_seconds, 30.0),
        _unit(features.note_density, 20.0),
        _unit(features.onset_density, 12.0),
        _unit(features.average_polyphony, 10.0),
        _unit(float(features.peak_polyphony), 16.0),
        _unit(features.average_chord_size, 10.0),
        _unit(features.pitch_mean - 21.0, 87.0),
        _unit(float(features.pitch_span), 87.0),
        _unit(features.velocity_mean, 127.0),
        _unit(float(features.velocity_range), 126.0),
        _unit(features.median_onset_gap, 2.0),
        _unit(features.median_note_duration, 4.0),
        1.0 if features.bass_and_treble else 0.0,
    )
    if len(vector) != len(CONTROL_FEATURE_NAMES):
        raise RuntimeError("Motif control vector does not match its declared feature names.")
    return vector
