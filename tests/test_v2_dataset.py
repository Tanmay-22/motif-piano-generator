from __future__ import annotations

import csv
import random

import pretty_midi
import pytest

from motifgen.tokenizer import RecordedNote
from motifgen.v2 import CompleteNoteTokenizer, EventType, MusicCategory
from training.v2_dataset import (
    COMPOSER_CATEGORIES,
    MaestroV2PhraseDataset,
    PhraseExtractionConfig,
    _window_to_pair,
    build_phrase_index,
    category_for_composer,
    extract_phrase_pair,
    load_v2_split_records,
)


def _phrase_notes() -> list[RecordedNote]:
    return [
        RecordedNote(48, 0.0, 0.4, 75),
        RecordedNote(60, 0.0, 0.4, 90),
        RecordedNote(62, 0.6, 1.0, 92),
        RecordedNote(64, 1.2, 1.6, 94),
        RecordedNote(65, 1.8, 2.2, 96),
        RecordedNote(67, 2.4, 2.8, 98),
        RecordedNote(69, 3.0, 3.4, 100),
        RecordedNote(71, 3.6, 4.0, 102),
        RecordedNote(72, 4.2, 4.6, 104),
        RecordedNote(74, 4.8, 5.2, 106),
        RecordedNote(76, 5.4, 5.8, 108),
        RecordedNote(77, 6.0, 6.4, 110),
        RecordedNote(79, 6.6, 7.0, 112),
    ]


def _small_config() -> PhraseExtractionConfig:
    return PhraseExtractionConfig(
        motif_min_seconds=2.0,
        motif_max_seconds=3.0,
        continuation_seconds=3.0,
        min_motif_notes=4,
        min_continuation_notes=4,
        max_motif_events=32,
        max_continuation_events=32,
    )


def _write_midi(path, notes: list[RecordedNote]) -> None:
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes = [
        pretty_midi.Note(note.velocity, note.pitch, note.start, note.end) for note in notes
    ]
    midi.instruments.append(instrument)
    midi.write(str(path))


def test_official_composer_categories_include_transcriptions():
    assert len(COMPOSER_CATEGORIES) == 43
    assert category_for_composer("Johann Sebastian Bach / Ferruccio Busoni") is MusicCategory.BAROQUE_CLASSICAL
    assert category_for_composer("Frédéric Chopin") is MusicCategory.ROMANTIC
    assert category_for_composer("Claude Debussy") is MusicCategory.IMPRESSIONIST_MODERN
    with pytest.raises(ValueError, match="No period category"):
        category_for_composer("Unknown Composer")


def test_phrase_pair_is_onset_aligned_and_preserves_transition_delay():
    tokenizer = CompleteNoteTokenizer()
    pair = extract_phrase_pair(
        _phrase_notes(),
        MusicCategory.ROMANTIC,
        tokenizer,
        _small_config(),
        random.Random(7),
    )
    assert pair.motif_events[0].event_type == EventType.BOS
    assert pair.motif_events[-1].event_type == EventType.SEP
    assert pair.continuation_events[0].event_type == EventType.BOS
    assert pair.continuation_events[-1].event_type == EventType.EOS
    assert pair.motif_note_count >= 4
    assert pair.continuation_note_count >= 4
    source_onsets = {note.start for note in _phrase_notes()}
    assert pair.source_start_seconds in source_onsets
    assert pair.split_seconds in source_onsets
    assert pair.motif_features.note_count == pair.motif_note_count

    continuation = tokenizer.events_to_notes(pair.continuation_events)
    source_last_motif_onset = max(
        note.start for note in _phrase_notes() if pair.source_start_seconds <= note.start < pair.split_seconds
    )
    assert continuation[0].start == pytest.approx(pair.split_seconds - source_last_motif_onset, abs=0.011)


def test_phrase_pair_clips_pedal_sustains_to_phrase_windows():
    notes = _phrase_notes()
    notes[0] = RecordedNote(
        notes[0].pitch,
        notes[0].start,
        45.0,
        notes[0].velocity,
    )
    continuation_index = next(
        index for index, note in enumerate(notes) if note.start == 2.4
    )
    continuation_note = notes[continuation_index]
    notes[continuation_index] = RecordedNote(
        continuation_note.pitch,
        continuation_note.start,
        50.0,
        continuation_note.velocity,
    )
    tokenizer = CompleteNoteTokenizer()
    config = _small_config()
    pair = _window_to_pair(
        build_phrase_index(notes),
        start=0.0,
        split=2.4,
        category=MusicCategory.ROMANTIC,
        tokenizer=tokenizer,
        config=config,
    )
    assert pair is not None

    motif = tokenizer.events_to_notes(pair.motif_events)
    continuation = tokenizer.events_to_notes(pair.continuation_events)
    assert max(note.end for note in motif) == pytest.approx(2.4, abs=0.011)
    # Continuation timing is relative to the final motif onset (1.8 seconds).
    assert max(note.end for note in continuation) == pytest.approx(3.6, abs=0.011)
    assert all(
        note.end - note.start <= tokenizer.max_time_seconds
        for note in [*motif, *continuation]
    )


def test_metadata_loader_preserves_official_split_and_category(tmp_path):
    midi_path = tmp_path / "2018" / "piece.midi"
    midi_path.parent.mkdir()
    _write_midi(midi_path, _phrase_notes())
    metadata = tmp_path / "maestro-v3.0.0.csv"
    with metadata.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "canonical_composer",
                "canonical_title",
                "split",
                "year",
                "midi_filename",
                "audio_filename",
                "duration",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "canonical_composer": "Claude Debussy",
                "canonical_title": "Test Piece",
                "split": "validation",
                "year": "2018",
                "midi_filename": "2018/piece.midi",
                "audio_filename": "",
                "duration": "7.0",
            }
        )
    records = load_v2_split_records(tmp_path, "validation")
    assert len(records) == 1
    assert records[0].category is MusicCategory.IMPRESSIONIST_MODERN
    assert records[0].midi_path == midi_path.resolve()


def test_v2_dataset_changes_training_window_by_epoch_and_drops_style(tmp_path):
    midi_path = tmp_path / "piece.midi"
    _write_midi(midi_path, _phrase_notes())
    from training.v2_dataset import MaestroV2Record

    record = MaestroV2Record(
        midi_path=midi_path,
        split="train",
        composer="Frédéric Chopin",
        title="Test",
        category=MusicCategory.ROMANTIC,
        performance_year=2018,
        duration_seconds=7.0,
    )
    dataset = MaestroV2PhraseDataset(
        [record],
        CompleteNoteTokenizer(),
        _small_config(),
        training=True,
        examples_per_piece=2,
        style_dropout_probability=1.0,
        seed=11,
    )
    first = dataset[0]
    dataset.set_epoch(1)
    second = dataset[0]
    assert first.conditioning_category is MusicCategory.AUTO
    assert second.conditioning_category is MusicCategory.AUTO
    assert len(dataset) == 2
    assert (first.source_start_seconds, first.split_seconds) != (
        second.source_start_seconds,
        second.split_seconds,
    )
