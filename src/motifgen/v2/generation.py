from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from motifgen.tokenizer import RecordedNote

from .config import V2GenerationConfig, V2ModelConfig
from .controls import category_id, motif_feature_vector, texture_id
from .features import MotifFeatures, TextureClass, extract_motif_features
from .model import FactorizedLogits, MotifContinuationTransformer
from .tokenizer import CompleteNoteEvent, CompleteNoteTokenizer, EventType, MusicCategory


@dataclass(frozen=True)
class V2GenerationResult:
    motif_notes: tuple[RecordedNote, ...]
    continuation_notes: tuple[RecordedNote, ...]
    continuation_events: tuple[CompleteNoteEvent, ...]
    category: MusicCategory
    motif_features: MotifFeatures
    target_seconds: int
    reached_target_duration: bool
    timed_out: bool

    @property
    def all_notes(self) -> list[RecordedNote]:
        return sorted(
            [*self.motif_notes, *self.continuation_notes],
            key=lambda note: (note.start, note.pitch, note.end),
        )

    @property
    def motif_end_seconds(self) -> float:
        return max((note.end for note in self.motif_notes), default=0.0)

    @property
    def continuation_duration_seconds(self) -> float:
        if not self.continuation_notes:
            return 0.0
        origin = max((note.start for note in self.motif_notes), default=0.0)
        return max(note.end for note in self.continuation_notes) - origin


class V2MotifGenerator:
    def __init__(
        self,
        model: MotifContinuationTransformer,
        tokenizer: CompleteNoteTokenizer,
        generation_config: V2GenerationConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.generation_config = generation_config or V2GenerationConfig()
        self.device = torch.device(device)
        self.model_kind = "v2"

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> "V2MotifGenerator":
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint.get("format_version") != 2:
            raise ValueError("This is not a v2 motif checkpoint.")
        tokenizer = CompleteNoteTokenizer(**checkpoint["tokenizer_config"])
        model = MotifContinuationTransformer(V2ModelConfig.from_dict(checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model_state"])
        return cls(model, tokenizer, device=device)

    @staticmethod
    def _normalize_motif(notes: Sequence[RecordedNote]) -> list[RecordedNote]:
        if len(notes) < 2:
            raise ValueError("Provide at least two notes for a motif.")
        ordered = sorted(notes, key=lambda note: (note.start, note.pitch, note.end))
        origin = ordered[0].start
        return [
            RecordedNote(
                pitch=note.pitch,
                start=max(0.0, note.start - origin),
                end=max(0.01, note.end - origin),
                velocity=note.velocity,
            )
            for note in ordered
        ]

    def _sample(
        self,
        logits: torch.Tensor,
        allowed_ids: Sequence[int],
        temperature: float,
        rng: torch.Generator,
    ) -> int:
        if not allowed_ids:
            raise ValueError("A generation constraint excluded every possible value.")
        values = logits.float() / temperature
        values = torch.nan_to_num(values, nan=-1e4, posinf=1e4, neginf=-1e4)
        allowed = torch.tensor(sorted(set(allowed_ids)), dtype=torch.long, device=values.device)
        allowed_values = values[allowed]
        top_k = min(self.generation_config.top_k, allowed_values.numel())
        top_values, top_positions = torch.topk(allowed_values, top_k)
        top_ids = allowed[top_positions]
        sorted_values, order = torch.sort(top_values, descending=True)
        sorted_ids = top_ids[order]
        probabilities = torch.softmax(sorted_values, dim=-1)
        cumulative = torch.cumsum(probabilities, dim=-1)
        remove = cumulative > self.generation_config.top_p
        if remove.numel() > 1:
            remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_values = sorted_values.masked_fill(remove, -torch.inf)
        probabilities = torch.softmax(sorted_values, dim=-1)
        choice = torch.multinomial(probabilities, 1, generator=rng)
        return int(sorted_ids[choice].item())

    def _sample_time_steps(
        self,
        logits: FactorizedLogits,
        prefix: str,
        temperature: float,
        rng: torch.Generator,
    ) -> int:
        coarse_values = getattr(logits, f"{prefix}_coarse")[0, -1]
        fine_values = getattr(logits, f"{prefix}_fine")[0, -1]
        coarse_id = self._sample(
            coarse_values,
            range(1, coarse_values.numel()),
            temperature,
            rng,
        )
        maximum_coarse_id = self.tokenizer.max_time_seconds + 1
        fine_ids = [1] if coarse_id == maximum_coarse_id else list(range(1, fine_values.numel()))
        fine_id = self._sample(fine_values, fine_ids, temperature, rng)
        return self.tokenizer.time_feature_ids_to_steps(coarse_id, fine_id)

    @staticmethod
    def _texture_limits(features: MotifFeatures) -> tuple[int, int, int]:
        if features.texture is TextureClass.MONOPHONIC:
            return 1, 1, 12
        if features.texture is TextureClass.LIGHT_POLYPHONIC:
            return min(5, max(2, features.peak_polyphony + 1)), min(
                5, max(2, round(features.average_chord_size) + 1)
            ), 18
        return min(10, max(4, features.peak_polyphony + 2)), min(
            10, max(4, round(features.average_chord_size) + 2)
        ), 24

    @torch.inference_mode()
    def generate(
        self,
        motif_notes: Sequence[RecordedNote],
        target_seconds: int = 10,
        temperature: float = 1.0,
        category: MusicCategory = MusicCategory.AUTO,
        seed: int | None = None,
    ) -> V2GenerationResult:
        if target_seconds not in self.generation_config.allowed_durations:
            raise ValueError(f"Duration must be one of {self.generation_config.allowed_durations}.")
        if not self.generation_config.min_temperature <= temperature <= self.generation_config.max_temperature:
            raise ValueError(
                f"Temperature must be between {self.generation_config.min_temperature} and "
                f"{self.generation_config.max_temperature}."
            )

        motif = self._normalize_motif(motif_notes)
        maximum_motif_notes = self.model.config.max_motif_events - 2
        if len(motif) > maximum_motif_notes:
            raise ValueError(f"The v2 model accepts at most {maximum_motif_notes} motif notes.")
        features = extract_motif_features(motif)
        motif_events = self.tokenizer.notes_to_events(motif, add_bos=True)
        motif_events.append(CompleteNoteEvent.special(EventType.SEP))
        motif_tensor = torch.tensor(
            [[event.as_tuple() for event in motif_events]], dtype=torch.long, device=self.device
        )
        categories = torch.tensor([category_id(category)], dtype=torch.long, device=self.device)
        textures = torch.tensor([texture_id(features.texture)], dtype=torch.long, device=self.device)
        controls = torch.tensor(
            [motif_feature_vector(features)], dtype=torch.float32, device=self.device
        )
        memory = self.model.encode_motif(motif_tensor, categories, textures, controls)

        rng = torch.Generator(device=self.device)
        if seed is None:
            rng.seed()
        else:
            rng.manual_seed(seed)
        decoder_events = [CompleteNoteEvent.special(EventType.BOS)]
        generated_events: list[CompleteNoteEvent] = []
        generated_onset_steps = 0
        active_notes: list[tuple[int, int]] = []
        notes_at_onset = 0
        maximum_polyphony, maximum_chord_size, pitch_expansion = self._texture_limits(features)
        pitch_low = max(self.tokenizer.PITCH_MIN, features.pitch_min - pitch_expansion)
        pitch_high = min(self.tokenizer.PITCH_MAX, features.pitch_max + pitch_expansion)
        motif_velocity_bin = round(features.velocity_mean / 127 * (self.tokenizer.VELOCITY_BINS - 1))
        velocity_spread = max(
            4,
            round(features.velocity_range / 127 * (self.tokenizer.VELOCITY_BINS - 1)) + 2,
        )
        velocity_low = max(0, motif_velocity_bin - velocity_spread)
        velocity_high = min(self.tokenizer.VELOCITY_BINS - 1, motif_velocity_bin + velocity_spread)
        median_gap_steps = max(1, round(features.median_onset_gap * self.tokenizer.sample_rate))
        maximum_gap_steps = min(
            self.tokenizer.max_time_steps,
            max(round(1.5 * self.tokenizer.sample_rate), median_gap_steps * 4),
        )
        median_duration_steps = max(
            1, round(features.median_note_duration * self.tokenizer.sample_rate)
        )
        minimum_duration_steps = max(3, round(median_duration_steps * 0.2))
        maximum_duration_steps = min(
            8 * self.tokenizer.sample_rate,
            max(round(1.5 * self.tokenizer.sample_rate), median_duration_steps * 3),
        )
        maximum_density = max(2.0, min(16.0, features.note_density * 1.75))
        target_steps = target_seconds * self.tokenizer.sample_rate
        deadline = time.monotonic() + self.generation_config.wall_clock_seconds
        timed_out = False
        maximum_notes = self.model.config.max_continuation_events - 2

        while len(generated_events) < maximum_notes:
            if time.monotonic() >= deadline:
                timed_out = True
                break
            decoder_tensor = torch.tensor(
                [[event.as_tuple() for event in decoder_events]],
                dtype=torch.long,
                device=self.device,
            )
            logits = self.model.decode(decoder_tensor, memory)
            delta_steps = min(
                self._sample_time_steps(logits, "delta", temperature, rng), maximum_gap_steps
            )
            if generated_events and delta_steps == 0:
                if notes_at_onset >= maximum_chord_size or maximum_chord_size == 1:
                    delta_steps = max(1, median_gap_steps // 4)

            proposed_onset = generated_onset_steps + delta_steps
            minimum_density_onset = round(
                max(0.0, (len(generated_events) + 1) / maximum_density - (1 / maximum_density))
                * self.tokenizer.sample_rate
            )
            proposed_onset = max(proposed_onset, minimum_density_onset)
            active_notes = [item for item in active_notes if item[1] > proposed_onset]
            while len(active_notes) >= maximum_polyphony:
                proposed_onset = max(proposed_onset, min(end for _, end in active_notes))
                active_notes = [item for item in active_notes if item[1] > proposed_onset]
            if proposed_onset >= target_steps and generated_events:
                break
            delta_steps = max(0, proposed_onset - generated_onset_steps)
            if delta_steps > 0:
                notes_at_onset = 0
            generated_onset_steps = proposed_onset

            active_pitches = {pitch for pitch, _ in active_notes}
            allowed_pitches = [
                pitch for pitch in range(pitch_low, pitch_high + 1) if pitch not in active_pitches
            ]
            pitch_id = self._sample(
                logits.pitch[0, -1],
                [
                    (pitch - self.tokenizer.PITCH_MIN) + 1
                    for pitch in allowed_pitches
                ],
                temperature,
                rng,
            )
            pitch = self.tokenizer.PITCH_MIN + pitch_id - 1
            duration_steps = self._sample_time_steps(logits, "duration", temperature, rng)
            duration_steps = max(minimum_duration_steps, min(maximum_duration_steps, duration_steps))
            duration_steps = min(duration_steps, max(5, target_steps - generated_onset_steps + 50))
            velocity_id = self._sample(
                logits.velocity[0, -1],
                [value + 1 for value in range(velocity_low, velocity_high + 1)],
                temperature,
                rng,
            )
            event = self.tokenizer.make_note_event(
                delta_steps=delta_steps,
                pitch=pitch,
                duration_steps=duration_steps,
                velocity_bin=velocity_id - 1,
            )
            generated_events.append(event)
            decoder_events.append(event)
            notes_at_onset += 1
            active_notes.append((pitch, generated_onset_steps + duration_steps))

        decoder_events.append(CompleteNoteEvent.special(EventType.EOS))
        relative_continuation = self.tokenizer.events_to_notes(decoder_events)
        motif_last_onset = max(note.start for note in motif)
        continuation = tuple(
            RecordedNote(
                pitch=note.pitch,
                start=note.start + motif_last_onset,
                end=note.end + motif_last_onset,
                velocity=note.velocity,
            )
            for note in relative_continuation
        )
        continuation_duration = max((note.end for note in relative_continuation), default=0.0)
        return V2GenerationResult(
            motif_notes=tuple(motif),
            continuation_notes=continuation,
            continuation_events=tuple(decoder_events),
            category=category,
            motif_features=features,
            target_seconds=target_seconds,
            reached_target_duration=continuation_duration >= target_seconds * 0.95,
            timed_out=timed_out,
        )
