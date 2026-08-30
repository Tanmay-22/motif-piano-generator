from __future__ import annotations

from motifgen.tokenizer import RecordedNote
from motifgen.v2 import CompleteNoteTokenizer, MusicCategory
from training.export_v2_examples import reference_notes, shifted_notes
from training.v2_dataset import PhraseExtractionConfig, extract_phrase_pair

import random


def _notes() -> list[RecordedNote]:
    return [
        RecordedNote(60 + (index % 5), index * 0.5, index * 0.5 + 0.3, 80)
        for index in range(24)
    ]


def test_reference_export_keeps_continuation_after_motif():
    tokenizer = CompleteNoteTokenizer()
    pair = extract_phrase_pair(
        _notes(),
        MusicCategory.ROMANTIC,
        tokenizer,
        PhraseExtractionConfig(
            motif_min_seconds=2,
            motif_max_seconds=3,
            continuation_seconds=3,
            min_motif_notes=4,
            min_continuation_notes=4,
            max_motif_events=32,
            max_continuation_events=32,
        ),
        random.Random(7),
    )
    motif, reference = reference_notes(pair, tokenizer)
    assert len(reference) > len(motif)
    motif_last_onset = max(note.start for note in motif)
    assert min(note.start for note in reference[len(motif) :]) >= motif_last_onset


def test_shifted_notes_preserve_expression():
    original = RecordedNote(64, 0.25, 0.75, 91)
    shifted = shifted_notes([original], 2.0)[0]
    assert shifted == RecordedNote(64, 2.25, 2.75, 91)
