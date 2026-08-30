from __future__ import annotations

import pytest

from motifgen.tokenizer import RecordedNote


def test_tokenizer_vocabulary_is_stable(tokenizer):
    assert tokenizer.vocab_size == 313
    assert tokenizer.pad_id == 0
    assert tokenizer.bos_id == 1
    assert tokenizer.sep_id == 3


def test_notes_round_trip_with_quantized_timing(tokenizer):
    notes = [
        RecordedNote(60, 0.00, 0.50, 96),
        RecordedNote(64, 0.25, 0.75, 110),
    ]
    tokens = tokenizer.notes_to_tokens(notes)
    restored = tokenizer.tokens_to_notes(tokens)
    assert [note.pitch for note in restored] == [60, 64]
    assert restored[0].start == pytest.approx(0.00, abs=0.011)
    assert restored[0].end == pytest.approx(0.50, abs=0.011)
    assert restored[1].start == pytest.approx(0.25, abs=0.011)
    assert restored[1].end == pytest.approx(0.75, abs=0.011)


def test_active_notes_are_closed_at_end(tokenizer):
    tokens = [tokenizer.token_to_id["VEL_20"], tokenizer.token_to_id["NOTE_ON_60"]]
    notes = tokenizer.tokens_to_notes(tokens)
    assert len(notes) == 1
    assert notes[0].end > notes[0].start


def test_prepare_motif_pads_without_special_token_leakage(tokenizer):
    motif = [tokenizer.token_to_id["VEL_20"], tokenizer.token_to_id["NOTE_ON_60"]]
    prepared, padding = tokenizer.prepare_motif(motif, max_tokens=4)
    assert prepared[:2] == motif
    assert prepared[2:] == [tokenizer.pad_id, tokenizer.pad_id]
    assert padding == [False, False, True, True]


def test_recorded_note_validation_rejects_bad_pitch():
    with pytest.raises(ValueError, match="between MIDI 21 and 108"):
        RecordedNote.from_mapping({"pitch": 5, "start": 0, "end": 1, "velocity": 100})

