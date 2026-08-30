from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from .config import DataConfig, GenerationConfig, ModelConfig
from .model import MusicTransformer
from .tokenizer import MidiTokenizer


@dataclass(frozen=True)
class GenerationResult:
    motif_tokens: list[int]
    continuation_tokens: list[int]
    duration_seconds: float
    reached_target_duration: bool

    @property
    def all_tokens(self) -> list[int]:
        return self.motif_tokens + self.continuation_tokens


class MotifGenerator:
    def __init__(
        self,
        model: MusicTransformer,
        tokenizer: MidiTokenizer,
        data_config: DataConfig,
        generation_config: GenerationConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.data_config = data_config
        self.generation_config = generation_config or GenerationConfig()
        self.device = torch.device(device)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> "MotifGenerator":
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model_config = ModelConfig.from_dict(checkpoint["model_config"])
        data_config = DataConfig.from_dict(checkpoint.get("data_config", {}))
        tokenizer = MidiTokenizer(
            sample_rate=data_config.sample_rate,
            max_time_shift_steps=data_config.max_time_shift_steps,
        )
        if tokenizer.vocab_size != model_config.vocab_size:
            raise ValueError("Checkpoint vocabulary does not match the deployed tokenizer.")
        model = MusicTransformer(model_config)
        model.load_state_dict(checkpoint["model_state"])
        return cls(model, tokenizer, data_config, device=device)

    @torch.inference_mode()
    def generate(
        self,
        motif_tokens: Sequence[int],
        target_seconds: int = 10,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> GenerationResult:
        if target_seconds not in self.generation_config.allowed_durations:
            raise ValueError(f"Duration must be one of {self.generation_config.allowed_durations}.")
        if not self.generation_config.min_temperature <= temperature <= self.generation_config.max_temperature:
            raise ValueError(
                f"Temperature must be between {self.generation_config.min_temperature} and "
                f"{self.generation_config.max_temperature}."
            )

        prepared_motif, motif_padding = self.tokenizer.prepare_motif(
            motif_tokens,
            max_tokens=self.data_config.motif_max_tokens,
        )
        prompt = [self.tokenizer.bos_id, *prepared_motif, self.tokenizer.sep_id]
        prompt_padding = [False, *motif_padding, False]
        sequence = list(prompt)
        padding = list(prompt_padding)
        continuation: list[int] = []
        elapsed = 0.0
        rng = torch.Generator(device=self.device)
        if seed is not None:
            rng.manual_seed(seed)
        else:
            rng.seed()

        for _ in range(self.generation_config.max_generated_tokens):
            window_tokens = sequence[-self.model.config.max_sequence_length :]
            window_padding = padding[-self.model.config.max_sequence_length :]
            token_tensor = torch.tensor([window_tokens], dtype=torch.long, device=self.device)
            padding_tensor = torch.tensor([window_padding], dtype=torch.bool, device=self.device)
            logits = self.model(token_tensor, padding_tensor)[0, -1].float() / temperature
            for token_id in self.tokenizer.forbidden_generation_ids:
                logits[token_id] = -torch.inf
            top_k = min(self.generation_config.top_k, logits.numel())
            top_values, top_indices = torch.topk(logits, top_k)
            probabilities = torch.softmax(top_values, dim=-1)
            sampled_index = torch.multinomial(probabilities, 1, generator=rng)
            next_token = int(top_indices[sampled_index].item())
            sequence.append(next_token)
            padding.append(False)
            continuation.append(next_token)
            token_name = self.tokenizer.id_to_token[next_token]
            if token_name.startswith("TIME_SHIFT_"):
                elapsed += int(token_name.rsplit("_", 1)[1]) / self.tokenizer.sample_rate
            if elapsed >= target_seconds:
                break

        clean_motif = [token for token in prepared_motif if token != self.tokenizer.pad_id]
        return GenerationResult(
            motif_tokens=clean_motif,
            continuation_tokens=continuation,
            duration_seconds=elapsed,
            reached_target_duration=elapsed >= target_seconds,
        )
