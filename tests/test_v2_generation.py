from __future__ import annotations

import torch
from torch import nn

from motifgen.tokenizer import RecordedNote
from motifgen.v2 import (
    CompleteNoteTokenizer,
    FactorizedLogits,
    MotifContinuationTransformer,
    MotifMemory,
    MusicCategory,
    V2GenerationConfig,
    V2ModelConfig,
    V2MotifGenerator,
)


class ScriptedFactorizedModel(nn.Module):
    def __init__(self, tokenizer: CompleteNoteTokenizer) -> None:
        super().__init__()
        self.config = V2ModelConfig.from_tokenizer(
            tokenizer,
            model_dim=32,
            heads=4,
            encoder_layers=1,
            decoder_layers=1,
            feedforward_dim=64,
            dropout=0,
            max_motif_events=40,
            max_continuation_events=40,
        )
        self.anchor = nn.Parameter(torch.zeros(1))

    def encode_motif(self, motif_events, category_ids, texture_ids, motif_controls):
        batch = motif_events.size(0)
        return MotifMemory(
            hidden=torch.zeros(batch, 1, self.config.model_dim, device=motif_events.device),
            padding_mask=torch.zeros(batch, 1, dtype=torch.bool, device=motif_events.device),
        )

    def decode(self, continuation_events, memory):
        batch, length, _ = continuation_events.shape

        def values(size, preferred):
            result = torch.full(
                (batch, length, size), -20.0, device=continuation_events.device
            )
            result[..., preferred] = 20.0
            return result

        return FactorizedLogits(
            event_type=values(self.config.event_type_vocab_size, 2),
            delta_coarse=values(self.config.delta_coarse_vocab_size, 1),
            delta_fine=values(self.config.delta_fine_vocab_size, 1),
            pitch=values(self.config.pitch_vocab_size, (60 - 21) + 1),
            duration_coarse=values(self.config.duration_coarse_vocab_size, 1),
            duration_fine=values(self.config.duration_fine_vocab_size, 81),
            velocity=values(self.config.velocity_vocab_size, 24),
        )


def _monophonic_motif() -> list[RecordedNote]:
    return [
        RecordedNote(60, 0.0, 0.4, 88),
        RecordedNote(62, 0.5, 0.9, 92),
        RecordedNote(64, 1.0, 1.4, 96),
    ]


def _peak_polyphony(notes: list[RecordedNote] | tuple[RecordedNote, ...]) -> int:
    changes = [
        item
        for note in notes
        for item in ((round(note.start, 6), 1), (round(note.end, 6), -1))
    ]
    active = peak = 0
    for _, change in sorted(changes, key=lambda item: (item[0], item[1])):
        active += change
        peak = max(peak, active)
    return peak


def test_constrained_generation_preserves_monophonic_texture_and_pitch_range():
    tokenizer = CompleteNoteTokenizer()
    generator = V2MotifGenerator(
        ScriptedFactorizedModel(tokenizer),
        tokenizer,
        V2GenerationConfig(wall_clock_seconds=5),
    )
    result = generator.generate(
        _monophonic_motif(),
        target_seconds=5,
        temperature=1.0,
        category=MusicCategory.ROMANTIC,
        seed=3,
    )
    assert result.category is MusicCategory.ROMANTIC
    assert result.continuation_notes
    assert result.reached_target_duration
    assert result.timed_out is False
    assert all(48 <= note.pitch <= 76 for note in result.continuation_notes)
    ordered = sorted(result.continuation_notes, key=lambda note: note.start)
    assert all(first.end <= second.start + 1e-6 for first, second in zip(ordered, ordered[1:]))


def test_polyphonic_motif_allows_bounded_chords_without_duplicate_active_pitches():
    tokenizer = CompleteNoteTokenizer()
    generator = V2MotifGenerator(
        ScriptedFactorizedModel(tokenizer),
        tokenizer,
        V2GenerationConfig(wall_clock_seconds=5),
    )
    motif = [
        RecordedNote(48, 0.0, 0.5, 82),
        RecordedNote(64, 0.0, 0.5, 94),
        RecordedNote(50, 0.6, 1.1, 84),
        RecordedNote(67, 0.6, 1.1, 96),
        RecordedNote(52, 1.2, 1.7, 86),
        RecordedNote(69, 1.2, 1.7, 98),
    ]
    result = generator.generate(motif, target_seconds=5, seed=9)
    assert 2 <= _peak_polyphony(result.continuation_notes) <= 3
    for index, note in enumerate(result.continuation_notes):
        assert not any(
            other.pitch == note.pitch
            and other.start < note.end - 1e-6
            and note.start < other.end - 1e-6
            for other in result.continuation_notes[index + 1 :]
        )


def test_v2_checkpoint_auto_loader_uses_format_two(tmp_path):
    tokenizer = CompleteNoteTokenizer()
    config = V2ModelConfig.from_tokenizer(
        tokenizer,
        model_dim=32,
        heads=4,
        encoder_layers=1,
        decoder_layers=1,
        feedforward_dim=64,
        max_motif_events=40,
        max_continuation_events=40,
    )
    model = MotifContinuationTransformer(config)
    path = tmp_path / "v2.pt"
    torch.save(
        {
            "format_version": 2,
            "model_kind": "motif_encoder_decoder_v2",
            "model_state": model.state_dict(),
            "model_config": config.to_dict(),
            "tokenizer_config": {"sample_rate": 100, "max_time_seconds": 30},
        },
        path,
    )
    restored = V2MotifGenerator.from_checkpoint(path)
    assert restored.model_kind == "v2"
    assert restored.model.config.model_dim == 32
