from __future__ import annotations

import random

import pretty_midi
import pytest
import torch
from torch.utils.data import DataLoader

from motifgen.tokenizer import RecordedNote
from motifgen.v2 import (
    CompleteNoteTokenizer,
    MotifContinuationTransformer,
    MusicCategory,
    V2ModelConfig,
)
from training.train_v2 import (
    TrainingProgress,
    evaluation_batch_limit,
    evaluate_v2,
    learning_rate_multiplier,
    make_scheduler,
    restore_training_checkpoint,
    training_checkpoint,
)
from training.v2_batching import collate_v2_phrase_pairs
from training.v2_dataset import (
    MaestroV2Record,
    PhraseExtractionConfig,
    extract_phrase_pair,
    load_or_create_note_cache,
)


def _notes() -> list[RecordedNote]:
    return [
        RecordedNote(48, 0.0, 0.4, 75),
        RecordedNote(60, 0.0, 0.4, 90),
        *[
            RecordedNote(60 + index, index * 0.6, index * 0.6 + 0.4, 80 + index)
            for index in range(1, 13)
        ],
    ]


def _write_midi(path) -> None:
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes = [
        pretty_midi.Note(note.velocity, note.pitch, note.start, note.end) for note in _notes()
    ]
    midi.instruments.append(instrument)
    midi.write(str(path))


def _phrase_config() -> PhraseExtractionConfig:
    return PhraseExtractionConfig(
        motif_min_seconds=2,
        motif_max_seconds=3,
        continuation_seconds=3,
        min_motif_notes=4,
        min_continuation_notes=4,
        max_motif_events=32,
        max_continuation_events=32,
    )


def _small_model() -> MotifContinuationTransformer:
    return MotifContinuationTransformer(
        V2ModelConfig.from_tokenizer(
            CompleteNoteTokenizer(),
            model_dim=32,
            heads=4,
            encoder_layers=1,
            decoder_layers=1,
            feedforward_dim=64,
            dropout=0,
            max_motif_events=40,
            max_continuation_events=40,
        )
    )


def test_note_cache_is_single_portable_archive(tmp_path):
    midi_path = tmp_path / "piece.midi"
    _write_midi(midi_path)
    record = MaestroV2Record(
        midi_path=midi_path,
        split="train",
        composer="Frédéric Chopin",
        title="Test",
        category=MusicCategory.ROMANTIC,
        performance_year=2018,
        duration_seconds=7.5,
    )
    cache_path = tmp_path / "cache" / "train-notes.npz"
    first = load_or_create_note_cache([record], CompleteNoteTokenizer(), cache_path)
    second = load_or_create_note_cache([record], CompleteNoteTokenizer(), cache_path)
    assert cache_path.exists()
    assert len(first[0]) == len(second[0]) == len(_notes())
    assert second[0][3].start == pytest.approx(first[0][3].start, abs=1e-5)


def test_warmup_and_cosine_schedule_have_safe_floor():
    assert learning_rate_multiplier(0, 100, 1000) == pytest.approx(0.01)
    assert learning_rate_multiplier(99, 100, 1000) == pytest.approx(1.0)
    assert learning_rate_multiplier(1000, 100, 1000) == pytest.approx(0.1)


def test_zero_evaluation_batch_limit_means_full_split():
    assert evaluation_batch_limit(0) is None
    assert evaluation_batch_limit(25) == 25
    with pytest.raises(ValueError, match="cannot be negative"):
        evaluation_batch_limit(-1)


def test_training_checkpoint_restores_full_progress_and_optimizer(tmp_path):
    tokenizer = CompleteNoteTokenizer()
    model = _small_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = make_scheduler(optimizer, 2, 10)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    progress = TrainingProgress(
        global_step=3,
        epoch=1,
        next_batch_index=7,
        best_validation_loss=2.5,
        history=[{"step": 3, "loss": 2.5}],
    )
    checkpoint = training_checkpoint(
        model, tokenizer, _phrase_config(), optimizer, scheduler, scaler, progress
    )
    path = tmp_path / "latest.pt"
    torch.save(checkpoint, path)

    restored_model = _small_model()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored_scheduler = make_scheduler(restored_optimizer, 2, 10)
    restored_scaler = torch.amp.GradScaler("cuda", enabled=False)
    loaded = torch.load(path, weights_only=False)
    restored = restore_training_checkpoint(
        loaded,
        restored_model,
        restored_optimizer,
        restored_scheduler,
        restored_scaler,
    )
    assert restored.global_step == 3
    assert restored.next_batch_index == 7
    assert restored.history == [{"step": 3, "loss": 2.5}]
    for first, second in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(first, second)


def test_evaluation_reports_correct_and_shuffled_motif_loss():
    tokenizer = CompleteNoteTokenizer()
    pairs = [
        extract_phrase_pair(
            _notes(),
            MusicCategory.ROMANTIC,
            tokenizer,
            _phrase_config(),
            random.Random(seed),
        )
        for seed in (2, 8)
    ]
    loader = DataLoader(pairs, batch_size=2, collate_fn=collate_v2_phrase_pairs)
    metrics = evaluate_v2(_small_model().eval(), loader, torch.device("cpu"))
    assert float(metrics["loss"]) > 0
    assert float(metrics["shuffled_motif_loss"]) > 0
    assert "motif_dependency_gap" in metrics
