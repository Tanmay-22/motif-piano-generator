from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from motifgen.tokenizer import RecordedNote
from motifgen.v2 import CompleteNoteTokenizer, MusicCategory, V2MotifGenerator
from training.download_data import download_maestro
from training.v2_dataset import (
    MaestroV2PhraseDataset,
    PhrasePair,
    PhraseExtractionConfig,
    load_v2_split_records,
)


EXAMPLE_CATEGORIES = (
    MusicCategory.BAROQUE_CLASSICAL,
    MusicCategory.ROMANTIC,
    MusicCategory.IMPRESSIONIST_MODERN,
)


def shifted_notes(notes: Sequence[RecordedNote], offset: float) -> list[RecordedNote]:
    return [
        RecordedNote(
            pitch=note.pitch,
            start=note.start + offset,
            end=note.end + offset,
            velocity=note.velocity,
        )
        for note in notes
    ]


def reference_notes(
    pair: PhrasePair, tokenizer: CompleteNoteTokenizer
) -> tuple[list[RecordedNote], list[RecordedNote]]:
    """Decode a fixed held-out motif and its true contiguous continuation."""

    motif = tokenizer.events_to_notes(pair.motif_events)
    continuation = tokenizer.events_to_notes(pair.continuation_events)
    continuation_origin = max((note.start for note in motif), default=0.0)
    return motif, [*motif, *shifted_notes(continuation, continuation_origin)]


def fixed_example_indices(dataset: MaestroV2PhraseDataset, count: int) -> list[int]:
    if count < 1:
        raise ValueError("Example count must be positive.")
    selected: list[int] = []
    for category in EXAMPLE_CATEGORIES:
        for index, (record, _) in enumerate(dataset.pieces):
            if record.category is category:
                selected.append(index)
                break
        if len(selected) >= count:
            return selected
    for index in range(len(dataset.pieces)):
        if index not in selected:
            selected.append(index)
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise ValueError(f"Only {len(selected)} held-out performances can produce examples.")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export deterministic MAESTRO test motifs, references, and v2 generations."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/v2-cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v2/examples"))
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--duration-seconds", type=int, choices=(5, 10, 20), default=10)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive.")
    if not 0.6 <= args.temperature <= 1.4:
        parser.error("temperature must be between 0.6 and 1.4.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available.")
    return args


def main() -> None:
    args = parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 2:
        raise ValueError("Example export requires a v2 inference checkpoint.")
    tokenizer = CompleteNoteTokenizer(**checkpoint["tokenizer_config"])
    phrase_config = PhraseExtractionConfig(**checkpoint["phrase_config"])
    generator = V2MotifGenerator.from_checkpoint(args.checkpoint, device=device)

    data_root = download_maestro(args.data_dir)
    records = load_v2_split_records(data_root, "test")
    dataset = MaestroV2PhraseDataset(
        records,
        tokenizer,
        phrase_config,
        training=False,
        examples_per_piece=1,
        seed=args.seed,
        cache_path=args.cache_dir / "test-notes.npz",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "checkpoint": args.checkpoint.name,
        "split": "test",
        "seed": args.seed,
        "duration_seconds": args.duration_seconds,
        "temperature": args.temperature,
        "examples": [],
    }
    for example_number, dataset_index in enumerate(
        fixed_example_indices(dataset, args.count), start=1
    ):
        pair = dataset[dataset_index]
        record, _ = dataset.pieces[dataset_index]
        motif, reference = reference_notes(pair, tokenizer)
        generated = generator.generate(
            motif,
            target_seconds=args.duration_seconds,
            temperature=args.temperature,
            category=pair.source_category,
            seed=args.seed + example_number,
        )
        prefix = f"example-{example_number}-{pair.source_category.value.replace('_', '-')}"
        motif_name = f"{prefix}-motif.mid"
        reference_name = f"{prefix}-reference.mid"
        generated_name = f"{prefix}-generated.mid"
        (args.output_dir / motif_name).write_bytes(tokenizer.notes_to_midi_bytes(motif))
        (args.output_dir / reference_name).write_bytes(tokenizer.notes_to_midi_bytes(reference))
        (args.output_dir / generated_name).write_bytes(
            tokenizer.notes_to_midi_bytes(generated.all_notes)
        )
        manifest["examples"].append(
            {
                "number": example_number,
                "composer": record.composer,
                "title": record.title,
                "source_category": pair.source_category.value,
                "inferred_texture": pair.motif_features.texture.value,
                "motif_note_count": len(motif),
                "generated_note_count": len(generated.continuation_notes),
                "reached_target_duration": generated.reached_target_duration,
                "timed_out": generated.timed_out,
                "motif_file": motif_name,
                "reference_file": reference_name,
                "generated_file": generated_name,
            }
        )
        print(f"Exported {prefix}: {record.composer} — {record.title}")

    manifest_path = args.output_dir / "examples-v2.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Example manifest: {manifest_path}")


if __name__ == "__main__":
    main()
