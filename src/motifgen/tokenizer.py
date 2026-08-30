from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import pretty_midi


@dataclass(frozen=True)
class RecordedNote:
    pitch: int
    start: float
    end: float
    velocity: int = 100

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "RecordedNote":
        try:
            pitch = int(value["pitch"])
            start = float(value["start"])
            end = float(value["end"])
            velocity = int(value.get("velocity", 100))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Each note needs numeric pitch, start, end, and velocity values.") from exc

        if not 21 <= pitch <= 108:
            raise ValueError("Piano pitches must be between MIDI 21 and 108.")
        if start < 0 or end <= start:
            raise ValueError("Every note must end after it starts and use non-negative time.")
        if end > 30:
            raise ValueError("Recorded motifs may be at most 30 seconds long.")
        if not 1 <= velocity <= 127:
            raise ValueError("MIDI velocity must be between 1 and 127.")
        return cls(pitch=pitch, start=start, end=end, velocity=velocity)

    def to_dict(self) -> dict[str, int | float]:
        return {
            "pitch": self.pitch,
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "velocity": self.velocity,
        }


class MidiTokenizer:
    PAD = "<PAD>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    SEP = "<SEP>"

    def __init__(self, sample_rate: int = 100, max_time_shift_steps: int = 100) -> None:
        self.sample_rate = sample_rate
        self.max_time_shift_steps = max_time_shift_steps
        self.token_to_id: dict[str, int] = {self.PAD: 0, self.BOS: 1, self.EOS: 2, self.SEP: 3}
        self.id_to_token: dict[int, str] = {value: key for key, value in self.token_to_id.items()}
        index = 4

        for pitch in range(21, 109):
            index = self._add(f"NOTE_ON_{pitch}", index)
        for pitch in range(21, 109):
            index = self._add(f"NOTE_OFF_{pitch}", index)
        for steps in range(0, max_time_shift_steps + 1):
            index = self._add(f"TIME_SHIFT_{steps}", index)
        for velocity in range(32):
            index = self._add(f"VEL_{velocity}", index)

        self.vocab_size = index
        self.pad_id = self.token_to_id[self.PAD]
        self.bos_id = self.token_to_id[self.BOS]
        self.eos_id = self.token_to_id[self.EOS]
        self.sep_id = self.token_to_id[self.SEP]
        self.forbidden_generation_ids = {
            self.pad_id,
            self.bos_id,
            self.eos_id,
            self.sep_id,
            self.token_to_id["TIME_SHIFT_0"],
        }

    def _add(self, token: str, index: int) -> int:
        self.token_to_id[token] = index
        self.id_to_token[index] = token
        return index + 1

    def notes_to_tokens(self, notes: Sequence[RecordedNote]) -> list[int]:
        events: list[tuple[float, int, str, RecordedNote]] = []
        for note in notes:
            events.append((note.start, 1, "on", note))
            events.append((note.end, 0, "off", note))
        events.sort(key=lambda event: (event[0], event[1], event[3].pitch))

        tokens: list[int] = []
        current_time = 0.0
        for event_time, _, event_type, note in events:
            delta_steps = max(0, round((event_time - current_time) * self.sample_rate))
            self._append_time_shift(tokens, delta_steps)
            current_time = event_time
            if event_type == "on":
                velocity = min(31, max(0, round(note.velocity / 127 * 31)))
                tokens.append(self.token_to_id[f"VEL_{velocity}"])
                tokens.append(self.token_to_id[f"NOTE_ON_{note.pitch}"])
            else:
                tokens.append(self.token_to_id[f"NOTE_OFF_{note.pitch}"])
        return tokens

    def midi_bytes_to_tokens(self, midi_bytes: bytes) -> list[int]:
        try:
            midi = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
        except Exception as exc:
            raise ValueError("The uploaded file is not a readable MIDI file.") from exc
        return self.notes_to_tokens(self._notes_from_pretty_midi(midi))

    def midi_path_to_tokens(self, path: str) -> list[int]:
        try:
            midi = pretty_midi.PrettyMIDI(path)
        except Exception as exc:
            raise ValueError(f"Could not read MIDI file: {path}") from exc
        return self.notes_to_tokens(self._notes_from_pretty_midi(midi))

    @staticmethod
    def _notes_from_pretty_midi(midi: pretty_midi.PrettyMIDI) -> list[RecordedNote]:
        notes = [
            RecordedNote(pitch=note.pitch, start=note.start, end=note.end, velocity=note.velocity)
            for instrument in midi.instruments
            if not instrument.is_drum
            for note in instrument.notes
            if 21 <= note.pitch <= 108 and note.end > note.start
        ]
        notes.sort(key=lambda note: (note.start, note.pitch, note.end))
        if not notes:
            raise ValueError("The MIDI file does not contain playable piano-range notes.")
        return notes

    def _append_time_shift(self, tokens: list[int], steps: int) -> None:
        while steps > self.max_time_shift_steps:
            tokens.append(self.token_to_id[f"TIME_SHIFT_{self.max_time_shift_steps}"])
            steps -= self.max_time_shift_steps
        if steps > 0:
            tokens.append(self.token_to_id[f"TIME_SHIFT_{steps}"])

    def prepare_motif(self, tokens: Iterable[int], max_tokens: int = 64) -> tuple[list[int], list[bool]]:
        clean = [int(token) for token in tokens if int(token) not in {self.pad_id, self.bos_id, self.eos_id, self.sep_id}]
        if not any(self.id_to_token.get(token, "").startswith("NOTE_ON_") for token in clean):
            raise ValueError("The motif must contain at least one note-on event.")
        clean = clean[:max_tokens]
        padding = max_tokens - len(clean)
        prepared = clean + ([self.pad_id] * padding)
        mask = ([False] * len(clean)) + ([True] * padding)
        return prepared, mask

    def tokens_to_notes(self, tokens: Iterable[int], close_active: bool = True) -> list[RecordedNote]:
        current_time = 0.0
        current_velocity = 100
        active: dict[int, tuple[float, int]] = {}
        notes: list[RecordedNote] = []

        for token_id in tokens:
            token = self.id_to_token.get(int(token_id), "")
            if token.startswith("TIME_SHIFT_"):
                current_time += int(token.rsplit("_", 1)[1]) / self.sample_rate
            elif token.startswith("VEL_"):
                quantized = int(token.rsplit("_", 1)[1])
                current_velocity = max(1, round(quantized / 31 * 127))
            elif token.startswith("NOTE_ON_"):
                pitch = int(token.rsplit("_", 1)[1])
                if pitch in active:
                    start, velocity = active[pitch]
                    notes.append(RecordedNote(pitch, start, max(current_time, start + 0.01), velocity))
                active[pitch] = (current_time, current_velocity)
            elif token.startswith("NOTE_OFF_"):
                pitch = int(token.rsplit("_", 1)[1])
                if pitch in active:
                    start, velocity = active.pop(pitch)
                    notes.append(RecordedNote(pitch, start, max(current_time, start + 0.01), velocity))

        if close_active:
            closing_time = max(current_time, max((start for start, _ in active.values()), default=0) + 0.1)
            for pitch, (start, velocity) in active.items():
                notes.append(RecordedNote(pitch, start, max(closing_time, start + 0.01), velocity))
        return sorted(notes, key=lambda note: (note.start, note.pitch, note.end))

    def tokens_to_midi_bytes(self, tokens: Iterable[int]) -> bytes:
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0, name="Motif Piano")
        for note in self.tokens_to_notes(tokens):
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

    def elapsed_seconds(self, tokens: Iterable[int]) -> float:
        total_steps = 0
        for token_id in tokens:
            token = self.id_to_token.get(int(token_id), "")
            if token.startswith("TIME_SHIFT_"):
                total_steps += int(token.rsplit("_", 1)[1])
        return total_steps / self.sample_rate

