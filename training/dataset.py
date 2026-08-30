from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Literal, Sequence

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from motifgen.config import DataConfig
from motifgen.tokenizer import MidiTokenizer


IGNORE_INDEX = -100


def load_split_paths(data_root: Path, split: str, limit: int | None = None) -> list[Path]:
    metadata_files = list(data_root.rglob("maestro-v3.0.0.csv"))
    if not metadata_files:
        raise FileNotFoundError("MAESTRO metadata CSV was not found. Run training.download_data first.")
    metadata = metadata_files[0]
    with metadata.open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    paths = [(metadata.parent / row["midi_filename"]).resolve() for row in rows]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"MAESTRO metadata references missing MIDI file: {missing[0]}")
    return paths[:limit] if limit else paths


def build_conditioned_example(
    motif: Sequence[int],
    continuation: Sequence[int],
    tokenizer: MidiTokenizer,
    config: DataConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prepared_motif, motif_padding = tokenizer.prepare_motif(motif, config.motif_max_tokens)
    prompt = [tokenizer.bos_id, *prepared_motif, tokenizer.sep_id]
    inputs = prompt + list(continuation[:-1])
    labels = [IGNORE_INDEX] * len(inputs)
    separator_position = len(prompt) - 1
    labels[separator_position:] = list(continuation)
    padding_mask = [False, *motif_padding, False] + ([False] * (len(continuation) - 1))
    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(padding_mask, dtype=torch.bool),
    )


def build_baseline_example(
    continuation: Sequence[int], tokenizer: MidiTokenizer
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = [tokenizer.bos_id, *continuation[:-1]]
    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(continuation, dtype=torch.long),
        torch.zeros(len(inputs), dtype=torch.bool),
    )


class MaestroTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        midi_paths: Sequence[Path],
        tokenizer: MidiTokenizer,
        config: DataConfig,
        mode: Literal["conditioned", "baseline"],
        training: bool,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.mode = mode
        self.training = training
        minimum = config.motif_max_tokens + config.continuation_tokens
        self.sequences: list[list[int]] = []
        for path in tqdm(midi_paths, desc=f"Tokenizing {mode} {'train' if training else 'eval'} split"):
            try:
                tokens = tokenizer.midi_path_to_tokens(str(path))
            except ValueError as exc:
                print(f"Skipping {path.name}: {exc}")
                continue
            if len(tokens) >= minimum:
                self.sequences.append(tokens)
        if not self.sequences:
            raise ValueError("No MIDI files in this split were long enough to create an example.")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence = self.sequences[index]
        required = self.config.motif_max_tokens + self.config.continuation_tokens
        max_start = len(sequence) - required
        start = random.randint(0, max_start) if self.training else (index * 9973) % (max_start + 1)
        motif_length = (
            random.randint(self.config.motif_min_tokens, self.config.motif_max_tokens)
            if self.training
            else self.config.motif_max_tokens
        )
        motif_end = start + motif_length
        motif = sequence[start:motif_end]
        continuation = sequence[motif_end : motif_end + self.config.continuation_tokens]
        if len(continuation) < self.config.continuation_tokens:
            fallback_end = start + self.config.motif_max_tokens
            motif = sequence[start:fallback_end]
            continuation = sequence[fallback_end : fallback_end + self.config.continuation_tokens]
        if self.mode == "conditioned":
            return build_conditioned_example(motif, continuation, self.tokenizer, self.config)
        return build_baseline_example(continuation, self.tokenizer)

