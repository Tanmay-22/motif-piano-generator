from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from statistics import median
from typing import Sequence

from motifgen.tokenizer import RecordedNote


class TextureClass(StrEnum):
    MONOPHONIC = "monophonic"
    LIGHT_POLYPHONIC = "light_polyphonic"
    FULL_POLYPHONIC = "full_polyphonic"


@dataclass(frozen=True)
class MotifFeatures:
    texture: TextureClass
    duration_seconds: float
    note_count: int
    onset_count: int
    note_density: float
    onset_density: float
    average_polyphony: float
    peak_polyphony: int
    average_chord_size: float
    pitch_min: int
    pitch_max: int
    pitch_mean: float
    pitch_span: int
    velocity_mean: float
    velocity_range: int
    median_onset_gap: float
    median_note_duration: float
    bass_and_treble: bool


def extract_motif_features(notes: Sequence[RecordedNote]) -> MotifFeatures:
    """Measure musical texture without attempting to assign literal hands."""

    if not notes:
        raise ValueError("At least one note is required to extract motif features.")
    ordered = sorted(notes, key=lambda note: (note.start, note.pitch, note.end))
    if any(note.start < 0 or note.end <= note.start for note in ordered):
        raise ValueError("Motif notes must use valid non-negative timing.")

    start = min(note.start for note in ordered)
    end = max(note.end for note in ordered)
    duration = max(end - start, 0.01)
    onset_groups: dict[int, list[RecordedNote]] = {}
    for note in ordered:
        onset_key = round(note.start * 100)
        onset_groups.setdefault(onset_key, []).append(note)

    sweep: list[tuple[float, int]] = []
    for note in ordered:
        sweep.append((note.start, 1))
        sweep.append((note.end, -1))
    # End events precede start events at the same instant.
    sweep.sort(key=lambda item: (item[0], item[1]))
    active = 0
    peak_polyphony = 0
    for _, change in sweep:
        active += change
        peak_polyphony = max(peak_polyphony, active)

    sounding_seconds = sum(note.end - note.start for note in ordered)
    average_polyphony = sounding_seconds / duration
    chord_sizes = [len(group) for group in onset_groups.values()]
    onset_times = sorted(key / 100 for key in onset_groups)
    onset_gaps = [later - earlier for earlier, later in zip(onset_times, onset_times[1:])]
    note_durations = [note.end - note.start for note in ordered]
    pitches = [note.pitch for note in ordered]
    velocities = [note.velocity for note in ordered]

    if peak_polyphony <= 1:
        texture = TextureClass.MONOPHONIC
    elif peak_polyphony <= 4 and average_polyphony < 2.75:
        texture = TextureClass.LIGHT_POLYPHONIC
    else:
        texture = TextureClass.FULL_POLYPHONIC

    return MotifFeatures(
        texture=texture,
        duration_seconds=duration,
        note_count=len(ordered),
        onset_count=len(onset_groups),
        note_density=len(ordered) / duration,
        onset_density=len(onset_groups) / duration,
        average_polyphony=average_polyphony,
        peak_polyphony=peak_polyphony,
        average_chord_size=sum(chord_sizes) / len(chord_sizes),
        pitch_min=min(pitches),
        pitch_max=max(pitches),
        pitch_mean=sum(pitches) / len(pitches),
        pitch_span=max(pitches) - min(pitches),
        velocity_mean=sum(velocities) / len(velocities),
        velocity_range=max(velocities) - min(velocities),
        median_onset_gap=median(onset_gaps) if onset_gaps else 0.0,
        median_note_duration=median(note_durations),
        bass_and_treble=min(pitches) < 60 <= max(pitches) and (max(pitches) - min(pitches)) >= 19,
    )
