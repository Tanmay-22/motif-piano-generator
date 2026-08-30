from __future__ import annotations

import pretty_midi
import pytest

from motifgen.tokenizer import RecordedNote
from motifgen.v2 import (
    CompleteNoteEvent,
    CompleteNoteTokenizer,
    EventType,
    MusicCategory,
    TextureClass,
    extract_motif_features,
)


@pytest.fixture
def v2_tokenizer() -> CompleteNoteTokenizer:
    return CompleteNoteTokenizer()


def test_complete_notes_round_trip_chords_delays_and_durations(v2_tokenizer):
    notes = [
        RecordedNote(60, 0.23, 0.78, 93),
        RecordedNote(64, 0.23, 1.11, 108),
        RecordedNote(67, 1.37, 3.82, 74),
    ]
    events = v2_tokenizer.notes_to_events(notes, add_bos=True, add_eos=True)
    restored = v2_tokenizer.events_to_notes(events)

    assert [event.event_type for event in events] == [
        EventType.BOS,
        EventType.NOTE,
        EventType.NOTE,
        EventType.NOTE,
        EventType.EOS,
    ]
    assert [note.pitch for note in restored] == [60, 64, 67]
    for actual, expected in zip(restored, notes):
        assert actual.start == pytest.approx(expected.start, abs=0.011)
        assert actual.end == pytest.approx(expected.end, abs=0.011)
        assert actual.velocity == pytest.approx(expected.velocity, abs=3)


def test_simultaneous_notes_have_zero_delta_after_first_note(v2_tokenizer):
    notes = [RecordedNote(48, 2.34, 3.0, 80), RecordedNote(72, 2.34, 4.5, 90)]
    events = v2_tokenizer.notes_to_events(notes)
    assert v2_tokenizer._join_time(events[0].delta_coarse - 1, events[0].delta_fine - 1) == 234
    assert v2_tokenizer._join_time(events[1].delta_coarse - 1, events[1].delta_fine - 1) == 0


def test_complete_note_representation_rejects_zero_duration(v2_tokenizer):
    invalid = CompleteNoteEvent(
        event_type=EventType.NOTE,
        delta_coarse=1,
        delta_fine=1,
        pitch=40,
        duration_coarse=1,
        duration_fine=1,
        velocity=20,
    )
    with pytest.raises(ValueError, match="positive duration"):
        v2_tokenizer.events_to_notes([invalid])


def test_midi_bytes_round_trip_uses_independent_complete_notes(v2_tokenizer):
    original = [
        RecordedNote(55, 0.0, 0.4, 64),
        RecordedNote(67, 0.0, 1.2, 96),
        RecordedNote(62, 0.6, 0.9, 88),
    ]
    midi_bytes = v2_tokenizer.events_to_midi_bytes(v2_tokenizer.notes_to_events(original))
    parsed = pretty_midi.PrettyMIDI(__import__("io").BytesIO(midi_bytes))
    restored = [note for instrument in parsed.instruments for note in instrument.notes]
    assert len(restored) == len(original)
    assert max(note.end for note in restored) == pytest.approx(1.2, abs=0.011)


def test_motif_features_distinguish_monophonic_and_two_register_polyphony():
    monophonic = [RecordedNote(60, 0.0, 0.4), RecordedNote(62, 0.5, 0.9)]
    polyphonic = [
        RecordedNote(43, 0.0, 1.0),
        RecordedNote(67, 0.0, 0.5),
        RecordedNote(71, 0.0, 0.5),
        RecordedNote(45, 0.5, 1.5),
        RecordedNote(69, 0.5, 1.0),
    ]

    mono_features = extract_motif_features(monophonic)
    poly_features = extract_motif_features(polyphonic)
    assert mono_features.texture is TextureClass.MONOPHONIC
    assert poly_features.texture is TextureClass.LIGHT_POLYPHONIC
    assert poly_features.bass_and_treble is True
    assert poly_features.peak_polyphony == 3
    assert poly_features.pitch_span == 28


def test_category_values_are_stable_and_validated():
    assert MusicCategory.from_value("Romantic") is MusicCategory.ROMANTIC
    assert [category.value for category in MusicCategory] == [
        "auto",
        "baroque_classical",
        "romantic",
        "impressionist_modern",
    ]
    with pytest.raises(ValueError, match="Category must be one of"):
        MusicCategory.from_value("jazz")
