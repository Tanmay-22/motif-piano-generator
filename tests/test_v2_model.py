from __future__ import annotations

import random

import pytest
import torch

from motifgen.tokenizer import RecordedNote
from motifgen.v2 import (
    CompleteNoteTokenizer,
    MotifContinuationTransformer,
    MusicCategory,
    V2ModelConfig,
    factorized_event_loss,
    motif_feature_vector,
)
from training.v2_batching import collate_v2_phrase_pairs
from training.v2_dataset import PhraseExtractionConfig, extract_phrase_pair


def _pair(transpose: int = 0):
    notes = [
        RecordedNote(48 + transpose, 0.0, 0.35, 70),
        RecordedNote(60 + transpose, 0.0, 0.45, 90),
        RecordedNote(62 + transpose, 0.6, 1.0, 92),
        RecordedNote(64 + transpose, 1.2, 1.6, 94),
        RecordedNote(65 + transpose, 1.8, 2.2, 96),
        RecordedNote(67 + transpose, 2.4, 2.8, 98),
        RecordedNote(69 + transpose, 3.0, 3.4, 100),
        RecordedNote(71 + transpose, 3.6, 4.0, 102),
        RecordedNote(72 + transpose, 4.2, 4.6, 104),
        RecordedNote(74 + transpose, 4.8, 5.2, 106),
        RecordedNote(76 + transpose, 5.4, 5.8, 108),
        RecordedNote(77 + transpose, 6.0, 6.4, 110),
        RecordedNote(79 + transpose, 6.6, 7.0, 112),
    ]
    return extract_phrase_pair(
        notes,
        MusicCategory.ROMANTIC,
        CompleteNoteTokenizer(),
        PhraseExtractionConfig(
            motif_min_seconds=2,
            motif_max_seconds=3,
            continuation_seconds=3,
            min_motif_notes=4,
            min_continuation_notes=4,
            max_motif_events=32,
            max_continuation_events=32,
        ),
        random.Random(4),
    )


def _small_model() -> MotifContinuationTransformer:
    config = V2ModelConfig.from_tokenizer(
        CompleteNoteTokenizer(),
        model_dim=48,
        heads=4,
        encoder_layers=2,
        decoder_layers=2,
        feedforward_dim=96,
        dropout=0.0,
        max_motif_events=40,
        max_continuation_events=40,
    )
    return MotifContinuationTransformer(config).eval()


def test_default_v2_model_is_compact_enough_for_colab_and_render():
    model = MotifContinuationTransformer(V2ModelConfig.from_tokenizer(CompleteNoteTokenizer()))
    assert 3_000_000 <= model.parameter_count() <= 7_000_000


def test_v2_forward_and_factorized_loss_shapes():
    batch = collate_v2_phrase_pairs([_pair(), _pair(1)])
    model = _small_model()
    logits = model(
        batch.motif_events,
        batch.decoder_inputs,
        batch.category_ids,
        batch.texture_ids,
        batch.motif_controls,
        batch.motif_padding_mask,
        batch.decoder_padding_mask,
    )
    assert logits.event_type.shape[:2] == batch.decoder_inputs.shape[:2]
    assert logits.pitch.shape[-1] == CompleteNoteTokenizer().feature_sizes["pitch"]
    loss = factorized_event_loss(logits, batch.decoder_targets)
    assert torch.isfinite(loss.total)
    assert loss.valid_events > 0
    assert loss.note_events > 0
    # Proper small-weight initialization keeps initial CE near vocabulary scale,
    # instead of the extreme losses produced by the v1 tied embedding defaults.
    assert loss.total.item() < 12


def test_decoder_output_changes_when_motif_changes():
    first = collate_v2_phrase_pairs([_pair()])
    second = collate_v2_phrase_pairs([_pair(3)])
    model = _small_model()
    first_logits = model(
        first.motif_events,
        first.decoder_inputs,
        first.category_ids,
        first.texture_ids,
        first.motif_controls,
    ).pitch
    second_logits = model(
        second.motif_events,
        first.decoder_inputs,
        first.category_ids,
        first.texture_ids,
        first.motif_controls,
    ).pitch
    assert not torch.allclose(first_logits, second_logits)


def test_motif_padding_does_not_change_unpadded_decoder_output():
    batch = collate_v2_phrase_pairs([_pair()])
    model = _small_model()
    memory = model.encode_motif(
        batch.motif_events,
        batch.category_ids,
        batch.texture_ids,
        batch.motif_controls,
    )
    extra = torch.zeros(1, 3, 7, dtype=torch.long)
    padded_motif = torch.cat([batch.motif_events, extra], dim=1)
    padded_memory = model.encode_motif(
        padded_motif,
        batch.category_ids,
        batch.texture_ids,
        batch.motif_controls,
    )
    original = model.decode(batch.decoder_inputs, memory).event_type
    padded = model.decode(batch.decoder_inputs, padded_memory).event_type
    assert torch.allclose(original, padded, atol=1e-5)


def test_motif_control_vector_is_normalized():
    vector = motif_feature_vector(_pair().motif_features)
    assert len(vector) == V2ModelConfig().control_feature_dim
    assert all(0.0 <= value <= 1.0 for value in vector)
