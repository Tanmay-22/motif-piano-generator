from __future__ import annotations

import io
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Iterable, Sequence

import pretty_midi

from motifgen.tokenizer import RecordedNote


class EventType(IntEnum):
    """Event types used by the factorized v2 representation."""

    PAD = 0
    BOS = 1
    NOTE = 2
    SEP = 3
    EOS = 4


class MusicCategory(StrEnum):
    """MAESTRO-compatible categories exposed by the web application."""

    AUTO = "auto"
    BAROQUE_CLASSICAL = "baroque_classical"
    ROMANTIC = "romantic"
    IMPRESSIONIST_MODERN = "impressionist_modern"

    @classmethod
    def from_value(cls, value: str) -> "MusicCategory":
        try:
            return cls(value.strip().lower())
        except ValueError as exc:
            allowed = ", ".join(category.value for category in cls)
            raise ValueError(f"Category must be one of: {allowed}.") from exc


@dataclass(frozen=True)
class CompleteNoteEvent:
    """One complete note or a structural event.

    All feature ids reserve zero for structural events and padding. A NOTE stores
    onset delay and duration as a whole-second component plus a 10 ms remainder.
    This makes invalid or permanently active notes impossible to represent.
    """

    event_type: int
    delta_coarse: int = 0
    delta_fine: int = 0
    pitch: int = 0
    duration_coarse: int = 0
    duration_fine: int = 0
    velocity: int = 0

    @classmethod
    def special(cls, event_type: EventType) -> "CompleteNoteEvent":
        if event_type is EventType.NOTE:
            raise ValueError("NOTE events require musical feature values.")
        return cls(event_type=int(event_type))

    def as_tuple(self) -> tuple[int, int, int, int, int, int, int]:
        return (
            self.event_type,
            self.delta_coarse,
            self.delta_fine,
            self.pitch,
            self.duration_coarse,
            self.duration_fine,
            self.velocity,
        )


class CompleteNoteTokenizer:
    """Convert piano notes to independent, factorized note events.

    Timing uses ``sample_rate`` steps per second. At the default of 100, both
    onset delay and duration round-trip at 10 ms resolution. Feature id zero is
    reserved for non-note events, so every musical value is offset by one.
    """

    PITCH_MIN = 21
    PITCH_MAX = 108
    VELOCITY_BINS = 32

    def __init__(self, sample_rate: int = 100, max_time_seconds: int = 30) -> None:
        if sample_rate < 1:
            raise ValueError("sample_rate must be positive.")
        if max_time_seconds < 1:
            raise ValueError("max_time_seconds must be positive.")
        self.sample_rate = sample_rate
        self.max_time_seconds = max_time_seconds
        self.max_time_steps = sample_rate * max_time_seconds

    @property
    def feature_sizes(self) -> dict[str, int]:
        """Vocabulary sizes, including the reserved zero value."""

        return {
            "event_type": len(EventType),
            "delta_coarse": self.max_time_seconds + 2,
            "delta_fine": self.sample_rate + 1,
            "pitch": (self.PITCH_MAX - self.PITCH_MIN + 1) + 1,
            "duration_coarse": self.max_time_seconds + 2,
            "duration_fine": self.sample_rate + 1,
            "velocity": self.VELOCITY_BINS + 1,
        }

    def notes_to_events(
        self,
        notes: Sequence[RecordedNote],
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[CompleteNoteEvent]:
        ordered = sorted(notes, key=lambda note: (note.start, note.pitch, note.end, note.velocity))
        events: list[CompleteNoteEvent] = []
        if add_bos:
            events.append(CompleteNoteEvent.special(EventType.BOS))

        previous_onset = 0.0
        for note in ordered:
            self._validate_note(note)
            delta_steps = round((note.start - previous_onset) * self.sample_rate)
            duration_steps = max(1, round((note.end - note.start) * self.sample_rate))
            delta_steps = self._bounded_steps(delta_steps, "Onset delay")
            duration_steps = self._bounded_steps(duration_steps, "Note duration")
            delta_coarse, delta_fine = self._split_time(delta_steps)
            duration_coarse, duration_fine = self._split_time(duration_steps)
            velocity_bin = min(
                self.VELOCITY_BINS - 1,
                max(0, round(note.velocity / 127 * (self.VELOCITY_BINS - 1))),
            )
            events.append(
                CompleteNoteEvent(
                    event_type=int(EventType.NOTE),
                    delta_coarse=delta_coarse + 1,
                    delta_fine=delta_fine + 1,
                    pitch=(note.pitch - self.PITCH_MIN) + 1,
                    duration_coarse=duration_coarse + 1,
                    duration_fine=duration_fine + 1,
                    velocity=velocity_bin + 1,
                )
            )
            previous_onset = note.start

        if add_eos:
            events.append(CompleteNoteEvent.special(EventType.EOS))
        return events

    def events_to_notes(self, events: Iterable[CompleteNoteEvent]) -> list[RecordedNote]:
        current_onset_steps = 0
        notes: list[RecordedNote] = []
        for event in events:
            event_type = self._event_type(event.event_type)
            if event_type is not EventType.NOTE:
                self._validate_special_event(event)
                continue
            self._validate_note_event(event)
            delta_steps = self._join_time(event.delta_coarse - 1, event.delta_fine - 1)
            duration_steps = self._join_time(event.duration_coarse - 1, event.duration_fine - 1)
            current_onset_steps += delta_steps
            pitch = self.PITCH_MIN + event.pitch - 1
            velocity_bin = event.velocity - 1
            velocity = max(1, round(velocity_bin / (self.VELOCITY_BINS - 1) * 127))
            start = current_onset_steps / self.sample_rate
            end = start + max(1, duration_steps) / self.sample_rate
            notes.append(RecordedNote(pitch=pitch, start=start, end=end, velocity=velocity))
        return notes

    def make_note_event(
        self,
        *,
        delta_steps: int,
        pitch: int,
        duration_steps: int,
        velocity_bin: int,
    ) -> CompleteNoteEvent:
        delta_steps = self._bounded_steps(delta_steps, "Onset delay")
        duration_steps = self._bounded_steps(duration_steps, "Note duration")
        if duration_steps < 1:
            raise ValueError("Note duration must be positive.")
        if not self.PITCH_MIN <= pitch <= self.PITCH_MAX:
            raise ValueError("Piano pitches must be between MIDI 21 and 108.")
        if not 0 <= velocity_bin < self.VELOCITY_BINS:
            raise ValueError("Velocity bin is outside its vocabulary.")
        delta_coarse, delta_fine = self._split_time(delta_steps)
        duration_coarse, duration_fine = self._split_time(duration_steps)
        return CompleteNoteEvent(
            event_type=int(EventType.NOTE),
            delta_coarse=delta_coarse + 1,
            delta_fine=delta_fine + 1,
            pitch=(pitch - self.PITCH_MIN) + 1,
            duration_coarse=duration_coarse + 1,
            duration_fine=duration_fine + 1,
            velocity=velocity_bin + 1,
        )

    def time_feature_ids_to_steps(self, coarse_id: int, fine_id: int) -> int:
        if not 1 <= coarse_id < self.feature_sizes["delta_coarse"]:
            raise ValueError("Coarse time id is outside its vocabulary.")
        if not 1 <= fine_id < self.feature_sizes["delta_fine"]:
            raise ValueError("Fine time id is outside its vocabulary.")
        steps = self._join_time(coarse_id - 1, fine_id - 1)
        return self._bounded_steps(steps, "Time value")

    def midi_bytes_to_events(self, midi_bytes: bytes) -> list[CompleteNoteEvent]:
        return self.notes_to_events(self.midi_bytes_to_notes(midi_bytes))

    def midi_bytes_to_notes(self, midi_bytes: bytes) -> list[RecordedNote]:
        try:
            midi = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
        except Exception as exc:
            raise ValueError("The uploaded file is not a readable MIDI file.") from exc
        return self._notes_from_midi(midi)

    def midi_path_to_events(self, path: str | Path) -> list[CompleteNoteEvent]:
        return self.notes_to_events(self.midi_path_to_notes(path))

    def midi_path_to_notes(self, path: str | Path) -> list[RecordedNote]:
        try:
            midi = pretty_midi.PrettyMIDI(str(path))
        except Exception as exc:
            raise ValueError(f"Could not read MIDI file: {path}") from exc
        return self._notes_from_midi(midi)

    def events_to_midi_bytes(self, events: Iterable[CompleteNoteEvent]) -> bytes:
        return self.notes_to_midi_bytes(self.events_to_notes(events))

    def notes_to_midi_bytes(self, notes: Iterable[RecordedNote]) -> bytes:
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0, name="Motif Piano v2")
        for note in notes:
            piano.notes.append(
                pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start,
                    end=note.end,
                )
            )
        midi.instruments.append(piano)
        destination = io.BytesIO()
        midi.write(destination)
        return destination.getvalue()

    @staticmethod
    def _notes_from_midi(midi: pretty_midi.PrettyMIDI) -> list[RecordedNote]:
        notes = [
            RecordedNote(pitch=note.pitch, start=note.start, end=note.end, velocity=note.velocity)
            for instrument in midi.instruments
            if not instrument.is_drum
            for note in instrument.notes
            if CompleteNoteTokenizer.PITCH_MIN <= note.pitch <= CompleteNoteTokenizer.PITCH_MAX
            and note.end > note.start
        ]
        if not notes:
            raise ValueError("The MIDI file does not contain playable piano-range notes.")
        return sorted(notes, key=lambda note: (note.start, note.pitch, note.end))

    def _split_time(self, steps: int) -> tuple[int, int]:
        return divmod(steps, self.sample_rate)

    def _join_time(self, coarse: int, fine: int) -> int:
        return (coarse * self.sample_rate) + fine

    def _bounded_steps(self, steps: int, label: str) -> int:
        if steps < 0:
            raise ValueError(f"{label} cannot be negative.")
        if steps > self.max_time_steps:
            raise ValueError(f"{label} cannot exceed {self.max_time_seconds} seconds.")
        return steps

    @staticmethod
    def _event_type(value: int) -> EventType:
        try:
            return EventType(value)
        except ValueError as exc:
            raise ValueError(f"Unknown v2 event type: {value}.") from exc

    @staticmethod
    def _validate_note(note: RecordedNote) -> None:
        if not CompleteNoteTokenizer.PITCH_MIN <= note.pitch <= CompleteNoteTokenizer.PITCH_MAX:
            raise ValueError("Piano pitches must be between MIDI 21 and 108.")
        if note.start < 0 or note.end <= note.start:
            raise ValueError("Every note must end after it starts and use non-negative time.")
        if not 1 <= note.velocity <= 127:
            raise ValueError("MIDI velocity must be between 1 and 127.")

    def _validate_note_event(self, event: CompleteNoteEvent) -> None:
        sizes = self.feature_sizes
        for name in (
            "delta_coarse",
            "delta_fine",
            "pitch",
            "duration_coarse",
            "duration_fine",
            "velocity",
        ):
            value = getattr(event, name)
            if not 1 <= value < sizes[name]:
                raise ValueError(f"NOTE event {name} id is outside its vocabulary.")
        duration = self._join_time(event.duration_coarse - 1, event.duration_fine - 1)
        if duration < 1:
            raise ValueError("NOTE events must have a positive duration.")

    @staticmethod
    def _validate_special_event(event: CompleteNoteEvent) -> None:
        if any(event.as_tuple()[1:]):
            raise ValueError("Structural events cannot contain musical feature values.")
