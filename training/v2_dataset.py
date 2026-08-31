from __future__ import annotations

import bisect
import csv
import hashlib
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from motifgen.tokenizer import RecordedNote
from motifgen.v2 import (
    CompleteNoteEvent,
    CompleteNoteTokenizer,
    EventType,
    MotifFeatures,
    MusicCategory,
    extract_motif_features,
)


# MAESTRO's canonical composer value sometimes appends an arranger after " / ".
# Categories intentionally follow the original composer at the start of the value.
COMPOSER_CATEGORIES: dict[str, MusicCategory] = {
    "Alban Berg": MusicCategory.IMPRESSIONIST_MODERN,
    "Alexander Scriabin": MusicCategory.IMPRESSIONIST_MODERN,
    "Antonio Soler": MusicCategory.BAROQUE_CLASSICAL,
    "Carl Maria von Weber": MusicCategory.ROMANTIC,
    "César Franck": MusicCategory.ROMANTIC,
    "Charles Gounod": MusicCategory.ROMANTIC,
    "Claude Debussy": MusicCategory.IMPRESSIONIST_MODERN,
    "Domenico Scarlatti": MusicCategory.BAROQUE_CLASSICAL,
    "Edvard Grieg": MusicCategory.ROMANTIC,
    "Felix Mendelssohn": MusicCategory.ROMANTIC,
    "Franz Liszt": MusicCategory.ROMANTIC,
    "Franz Schubert": MusicCategory.ROMANTIC,
    "Frédéric Chopin": MusicCategory.ROMANTIC,
    "Fritz Kreisler": MusicCategory.ROMANTIC,
    "George Enescu": MusicCategory.IMPRESSIONIST_MODERN,
    "George Frideric Handel": MusicCategory.BAROQUE_CLASSICAL,
    "Georges Bizet": MusicCategory.ROMANTIC,
    "Giuseppe Verdi": MusicCategory.ROMANTIC,
    "Henry Purcell": MusicCategory.BAROQUE_CLASSICAL,
    "Isaac Albéniz": MusicCategory.IMPRESSIONIST_MODERN,
    "Jean-Philippe Rameau": MusicCategory.BAROQUE_CLASSICAL,
    "Johann Christian Fischer": MusicCategory.BAROQUE_CLASSICAL,
    "Johann Pachelbel": MusicCategory.BAROQUE_CLASSICAL,
    "Johann Sebastian Bach": MusicCategory.BAROQUE_CLASSICAL,
    "Johann Strauss": MusicCategory.ROMANTIC,
    "Johannes Brahms": MusicCategory.ROMANTIC,
    "Joseph Haydn": MusicCategory.BAROQUE_CLASSICAL,
    "Leoš Janáček": MusicCategory.IMPRESSIONIST_MODERN,
    "Ludwig van Beethoven": MusicCategory.BAROQUE_CLASSICAL,
    "Mikhail Glinka": MusicCategory.ROMANTIC,
    "Mily Balakirev": MusicCategory.ROMANTIC,
    "Modest Mussorgsky": MusicCategory.ROMANTIC,
    "Muzio Clementi": MusicCategory.BAROQUE_CLASSICAL,
    "Niccolò Paganini": MusicCategory.ROMANTIC,
    "Nikolai Medtner": MusicCategory.ROMANTIC,
    "Nikolai Rimsky-Korsakov": MusicCategory.ROMANTIC,
    "Orlando Gibbons": MusicCategory.BAROQUE_CLASSICAL,
    "Percy Grainger": MusicCategory.IMPRESSIONIST_MODERN,
    "Pyotr Ilyich Tchaikovsky": MusicCategory.ROMANTIC,
    "Richard Wagner": MusicCategory.ROMANTIC,
    "Robert Schumann": MusicCategory.ROMANTIC,
    "Sergei Rachmaninoff": MusicCategory.ROMANTIC,
    "Wolfgang Amadeus Mozart": MusicCategory.BAROQUE_CLASSICAL,
}


@dataclass(frozen=True)
class MaestroV2Record:
    midi_path: Path
    split: str
    composer: str
    title: str
    category: MusicCategory
    performance_year: int
    duration_seconds: float


@dataclass(frozen=True)
class PhraseExtractionConfig:
    motif_min_seconds: float = 2.0
    motif_max_seconds: float = 6.0
    continuation_seconds: float = 10.0
    min_motif_notes: int = 4
    min_continuation_notes: int = 8
    max_motif_events: int = 128
    max_continuation_events: int = 256
    boundary_tolerance_seconds: float = 0.01
    relaxed_boundary_polyphony: int = 2

    def __post_init__(self) -> None:
        if self.motif_min_seconds <= 0 or self.motif_max_seconds < self.motif_min_seconds:
            raise ValueError("Motif duration bounds are invalid.")
        if self.continuation_seconds <= 0:
            raise ValueError("Continuation duration must be positive.")
        if self.min_motif_notes < 1 or self.min_continuation_notes < 1:
            raise ValueError("Minimum note counts must be positive.")
        if self.max_motif_events < self.min_motif_notes:
            raise ValueError("max_motif_events cannot be smaller than min_motif_notes.")
        if self.max_continuation_events < self.min_continuation_notes:
            raise ValueError("max_continuation_events cannot be smaller than min_continuation_notes.")


@dataclass(frozen=True)
class PhrasePair:
    motif_events: tuple[CompleteNoteEvent, ...]
    continuation_events: tuple[CompleteNoteEvent, ...]
    motif_features: MotifFeatures
    source_category: MusicCategory
    conditioning_category: MusicCategory
    source_start_seconds: float
    split_seconds: float
    continuation_origin_seconds: float

    @property
    def motif_note_count(self) -> int:
        return sum(event.event_type == EventType.NOTE for event in self.motif_events)

    @property
    def continuation_note_count(self) -> int:
        return sum(event.event_type == EventType.NOTE for event in self.continuation_events)


@dataclass(frozen=True)
class PhraseIndex:
    notes: tuple[RecordedNote, ...]
    quantized_note_onsets: tuple[float, ...]
    onset_times: tuple[float, ...]
    active_counts: tuple[int, ...]


def category_for_composer(canonical_composer: str) -> MusicCategory:
    primary_composer = canonical_composer.split(" / ", 1)[0].strip()
    try:
        return COMPOSER_CATEGORIES[primary_composer]
    except KeyError as exc:
        raise ValueError(
            f"No period category is configured for MAESTRO composer {canonical_composer!r}."
        ) from exc


def load_v2_split_records(
    data_root: Path,
    split: str,
    limit: int | None = None,
) -> list[MaestroV2Record]:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test.")
    metadata_files = list(data_root.rglob("maestro-v3.0.0.csv"))
    if not metadata_files:
        raise FileNotFoundError("MAESTRO metadata CSV was not found. Run training.download_data first.")

    records: list[MaestroV2Record] = []
    metadata_path = metadata_files[0]
    with metadata_path.open(encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        required = {
            "canonical_composer",
            "canonical_title",
            "split",
            "year",
            "midi_filename",
            "duration",
        }
        if not rows.fieldnames or not required.issubset(rows.fieldnames):
            raise ValueError("MAESTRO metadata CSV is missing required columns.")
        for row in rows:
            if row["split"] != split:
                continue
            midi_path = (metadata_path.parent / row["midi_filename"]).resolve()
            if not midi_path.exists():
                raise FileNotFoundError(f"MAESTRO metadata references missing MIDI file: {midi_path}")
            records.append(
                MaestroV2Record(
                    midi_path=midi_path,
                    split=split,
                    composer=row["canonical_composer"],
                    title=row["canonical_title"],
                    category=category_for_composer(row["canonical_composer"]),
                    performance_year=int(row["year"]),
                    duration_seconds=float(row["duration"]),
                )
            )
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"MAESTRO split {split!r} contains no usable records.")
    return records


def _note_cache_fingerprint(
    records: Sequence[MaestroV2Record], tokenizer: CompleteNoteTokenizer
) -> str:
    digest = hashlib.sha256()
    digest.update(f"v2:{tokenizer.sample_rate}:{tokenizer.max_time_seconds}".encode())
    for record in records:
        stat = record.midi_path.stat()
        relative_name = "/".join(record.midi_path.parts[-2:])
        digest.update(
            f"{relative_name}:{stat.st_size}:{record.composer}:{record.split}".encode("utf-8")
        )
    return digest.hexdigest()


def load_or_create_note_cache(
    records: Sequence[MaestroV2Record],
    tokenizer: CompleteNoteTokenizer,
    cache_path: Path,
) -> list[tuple[RecordedNote, ...]]:
    """Load all notes from one portable compressed archive, creating it if needed."""

    fingerprint = _note_cache_fingerprint(records, tokenizer)
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                cached_fingerprint = str(cached["fingerprint"].item())
                values = cached["notes"]
                offsets = cached["offsets"]
            if cached_fingerprint == fingerprint and len(offsets) == len(records) + 1:
                return [
                    tuple(
                        RecordedNote(
                            pitch=int(row[0]),
                            start=float(row[1]),
                            end=float(row[2]),
                            velocity=int(row[3]),
                        )
                        for row in values[offsets[index] : offsets[index + 1]]
                    )
                    for index in range(len(records))
                ]
        except (KeyError, OSError, ValueError):
            pass

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    all_notes: list[list[float]] = []
    offsets = [0]
    sequences: list[tuple[RecordedNote, ...]] = []
    for record in tqdm(records, desc=f"Caching v2 {records[0].split if records else 'empty'} notes"):
        notes = tuple(tokenizer.midi_path_to_notes(record.midi_path))
        sequences.append(notes)
        all_notes.extend(
            [float(note.pitch), note.start, note.end, float(note.velocity)] for note in notes
        )
        offsets.append(len(all_notes))
    values = np.asarray(all_notes, dtype=np.float32).reshape(-1, 4)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            fingerprint=np.asarray(fingerprint),
            notes=values,
            offsets=np.asarray(offsets, dtype=np.int64),
        )
    temporary.replace(cache_path)
    return sequences


def _onset_boundaries(
    notes: Sequence[RecordedNote],
    tolerance: float,
) -> tuple[list[float], list[int]]:
    """Return unique onset times and notes already active at each onset."""

    onset_groups: dict[int, list[RecordedNote]] = {}
    for note in notes:
        onset_groups.setdefault(round(note.start * 100), []).append(note)
    onset_times = [key / 100 for key in sorted(onset_groups)]

    active_ends: list[float] = []
    active_counts: list[int] = []
    for onset in onset_times:
        active_ends = [end for end in active_ends if end > onset + tolerance]
        active_counts.append(len(active_ends))
        active_ends.extend(note.end for note in onset_groups[round(onset * 100)])
    return onset_times, active_counts


def build_phrase_index(
    notes: Sequence[RecordedNote], tolerance: float = 0.01
) -> PhraseIndex:
    ordered = tuple(sorted(notes, key=lambda note: (note.start, note.pitch, note.end)))
    onset_times, active_counts = _onset_boundaries(ordered, tolerance)
    return PhraseIndex(
        notes=ordered,
        quantized_note_onsets=tuple(round(note.start * 100) / 100 for note in ordered),
        onset_times=tuple(onset_times),
        active_counts=tuple(active_counts),
    )


def _normalized_window_note(
    note: RecordedNote,
    origin: float,
    window_end: float,
    tokenizer: CompleteNoteTokenizer,
) -> RecordedNote:
    """Normalize a note and close pedal-length sustains at the phrase boundary.

    MAESTRO note ends include sustain-pedal extension, so a small number of
    notes last longer than the tokenizer's maximum duration. A training phrase
    cannot represent sound beyond its own window anyway. Clipping to both the
    phrase boundary and the tokenizer limit preserves the onset and expression
    without discarding the entire performance.
    """

    start = max(0.0, note.start - origin)
    absolute_end = min(
        note.end,
        window_end,
        note.start + tokenizer.max_time_seconds,
    )
    return RecordedNote(
        pitch=note.pitch,
        start=start,
        end=max(start + (1 / tokenizer.sample_rate), absolute_end - origin),
        velocity=note.velocity,
    )


def _window_to_pair(
    index: PhraseIndex,
    start: float,
    split: float,
    category: MusicCategory,
    tokenizer: CompleteNoteTokenizer,
    config: PhraseExtractionConfig,
) -> PhrasePair | None:
    continuation_end = split + config.continuation_seconds
    motif_start_index = bisect.bisect_left(index.quantized_note_onsets, start)
    split_index = bisect.bisect_left(index.quantized_note_onsets, split)
    continuation_end_index = bisect.bisect_left(
        index.quantized_note_onsets, continuation_end
    )
    motif_source = index.notes[motif_start_index:split_index]
    continuation_source = index.notes[split_index:continuation_end_index]
    if not config.min_motif_notes <= len(motif_source) <= config.max_motif_events:
        return None
    if not config.min_continuation_notes <= len(continuation_source) <= config.max_continuation_events:
        return None

    motif_notes = [
        _normalized_window_note(note, start, split, tokenizer)
        for note in motif_source
    ]
    last_motif_onset = max(note.start for note in motif_source)
    continuation_notes = [
        _normalized_window_note(
            note,
            last_motif_onset,
            continuation_end,
            tokenizer,
        )
        for note in continuation_source
    ]
    motif_events = tokenizer.notes_to_events(motif_notes, add_bos=True)
    motif_events.append(CompleteNoteEvent.special(EventType.SEP))
    continuation_events = tokenizer.notes_to_events(continuation_notes, add_bos=True, add_eos=True)
    return PhrasePair(
        motif_events=tuple(motif_events),
        continuation_events=tuple(continuation_events),
        motif_features=extract_motif_features(motif_notes),
        source_category=category,
        conditioning_category=category,
        source_start_seconds=start,
        split_seconds=split,
        continuation_origin_seconds=last_motif_onset,
    )


def extract_phrase_pair(
    notes: Sequence[RecordedNote],
    category: MusicCategory,
    tokenizer: CompleteNoteTokenizer,
    config: PhraseExtractionConfig,
    rng: random.Random,
) -> PhrasePair:
    """Extract an onset-aligned motif and its true contiguous continuation."""

    if not notes:
        raise ValueError("Cannot extract a phrase pair from an empty performance.")
    index = build_phrase_index(notes, config.boundary_tolerance_seconds)
    return extract_phrase_pair_from_index(index, category, tokenizer, config, rng)


def extract_phrase_pair_from_index(
    index: PhraseIndex,
    category: MusicCategory,
    tokenizer: CompleteNoteTokenizer,
    config: PhraseExtractionConfig,
    rng: random.Random,
) -> PhrasePair:
    """Extract a pair using boundaries precomputed once for an entire performance."""

    onset_times = index.onset_times
    active_counts = index.active_counts
    if len(onset_times) < 2:
        raise ValueError("The performance has too few onset boundaries for a phrase pair.")

    start_candidates = [
        onset
        for onset, active in zip(onset_times, active_counts)
        if active == 0
        and onset + config.motif_min_seconds + config.continuation_seconds <= onset_times[-1]
    ]
    rng.shuffle(start_candidates)

    # Prefer boundaries with no sounding note from before the split. If a piece
    # is heavily legato, allow a small amount of overlap without cutting tokens.
    for maximum_overlap in (0, config.relaxed_boundary_polyphony):
        for start in start_candidates:
            lower = bisect.bisect_left(onset_times, start + config.motif_min_seconds)
            upper = bisect.bisect_right(onset_times, start + config.motif_max_seconds)
            split_candidates = [
                onset_times[index]
                for index in range(lower, upper)
                if active_counts[index] <= maximum_overlap
            ]
            rng.shuffle(split_candidates)
            for split in split_candidates:
                pair = _window_to_pair(index, start, split, category, tokenizer, config)
                if pair is not None:
                    return pair
    raise ValueError("No phrase-aligned motif/continuation window satisfies the configured limits.")


class MaestroV2PhraseDataset(Dataset[PhrasePair]):
    """MAESTRO phrase pairs with repeatable per-epoch random windows."""

    def __init__(
        self,
        records: Sequence[MaestroV2Record],
        tokenizer: CompleteNoteTokenizer,
        config: PhraseExtractionConfig,
        *,
        training: bool,
        examples_per_piece: int = 8,
        style_dropout_probability: float = 0.25,
        seed: int = 42,
        cache_path: Path | None = None,
    ) -> None:
        if examples_per_piece < 1:
            raise ValueError("examples_per_piece must be positive.")
        if not 0 <= style_dropout_probability <= 1:
            raise ValueError("style_dropout_probability must be between zero and one.")
        self.tokenizer = tokenizer
        self.config = config
        self.training = training
        self.examples_per_piece = examples_per_piece
        self.style_dropout_probability = style_dropout_probability if training else 0.0
        self.seed = seed
        self.epoch = 0
        self.pieces: list[tuple[MaestroV2Record, PhraseIndex]] = []

        note_sequences = (
            load_or_create_note_cache(records, tokenizer, cache_path)
            if cache_path is not None
            else [tuple(tokenizer.midi_path_to_notes(record.midi_path)) for record in records]
        )
        for record, notes in tqdm(
            zip(records, note_sequences),
            total=len(records),
            desc="Indexing v2 MAESTRO performances",
        ):
            try:
                index = build_phrase_index(notes, config.boundary_tolerance_seconds)
                extract_phrase_pair_from_index(
                    index, record.category, tokenizer, config, random.Random(seed)
                )
            except ValueError as exc:
                print(f"Skipping {record.midi_path.name}: {exc}")
                continue
            self.pieces.append((record, index))
        if not self.pieces:
            raise ValueError("No MAESTRO performances could produce a v2 phrase pair.")

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch cannot be negative.")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.pieces) * self.examples_per_piece

    def __getitem__(self, index: int) -> PhrasePair:
        piece_index = index % len(self.pieces)
        record, phrase_index = self.pieces[piece_index]
        epoch = self.epoch if self.training else 0
        rng = random.Random(self.seed + (epoch * 1_000_003) + index)
        pair = extract_phrase_pair_from_index(
            phrase_index, record.category, self.tokenizer, self.config, rng
        )
        if rng.random() < self.style_dropout_probability:
            pair = replace(pair, conditioning_category=MusicCategory.AUTO)
        return pair
