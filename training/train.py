from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from motifgen.config import DataConfig, ModelConfig
from motifgen.generation import MotifGenerator
from motifgen.model import MusicTransformer
from motifgen.tokenizer import MidiTokenizer
from training.dataset import IGNORE_INDEX, MaestroTokenDataset, load_split_paths
from training.download_data import download_maestro


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: MusicTransformer,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    model.train(optimizer is not None)
    total_loss = 0.0
    total_batches = 0
    context = torch.enable_grad() if optimizer is not None else torch.inference_mode()
    with context:
        for inputs, labels, padding_mask in tqdm(loader, leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            padding_mask = padding_mask.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs, padding_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += float(loss.item())
            total_batches += 1
    return total_loss / max(total_batches, 1)


def train_one_model(
    mode: str,
    datasets: dict[str, MaestroTokenDataset],
    model_config: ModelConfig,
    data_config: DataConfig,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
) -> dict[str, float | list[dict[str, float]]]:
    set_seed(seed)
    model = MusicTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    loaders = {
        split: DataLoader(dataset, batch_size=batch_size, shuffle=split == "train", num_workers=0)
        for split, dataset in datasets.items()
    }
    best_validation = float("inf")
    history: list[dict[str, float]] = []
    checkpoint_path = output_dir / f"{mode}-best.pt"

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, loaders["train"], criterion, device, optimizer)
        validation_loss = run_epoch(model, loaders["validation"], criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        print(f"{mode} epoch {epoch:03d}: train={train_loss:.4f} validation={validation_loss:.4f}")
        if validation_loss < best_validation:
            best_validation = validation_loss
            torch.save(
                {
                    "format_version": 1,
                    "mode": mode,
                    "model_state": model.state_dict(),
                    "model_config": model_config.to_dict(),
                    "data_config": data_config.to_dict(),
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "seed": seed,
                },
                checkpoint_path,
            )

    best = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(best["model_state"])
    test_loss = run_epoch(model, loaders["test"], criterion, device)
    return {"best_validation_loss": best_validation, "test_loss": test_loss, "history": history}


def write_examples(
    checkpoint: Path,
    test_dataset: MaestroTokenDataset,
    tokenizer: MidiTokenizer,
    output_dir: Path,
) -> None:
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(exist_ok=True)
    generator = MotifGenerator.from_checkpoint(checkpoint)
    for index, sequence in enumerate(test_dataset.sequences[:3], start=1):
        motif = sequence[: generator.data_config.motif_max_tokens]
        result = generator.generate(motif, target_seconds=10, temperature=1.0, seed=1000 + index)
        (examples_dir / f"example-{index}.mid").write_bytes(tokenizer.tokens_to_midi_bytes(result.all_tokens))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline and motif-conditioned MAESTRO Transformers.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", choices=("conditioned", "baseline", "both"), default="both")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_root = download_maestro(args.data_dir)
    tokenizer = MidiTokenizer()
    data_config = DataConfig()
    model_config = ModelConfig(vocab_size=tokenizer.vocab_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    split_paths = {
        split: load_split_paths(data_root, split, args.limit_per_split)
        for split in ("train", "validation", "test")
    }
    modes = ("baseline", "conditioned") if args.models == "both" else (args.models,)
    metrics: dict[str, object] = {
        "seed": args.seed,
        "device": str(device),
        "data_config": asdict(data_config),
        "model_config": asdict(model_config),
        "split_file_counts": {key: len(value) for key, value in split_paths.items()},
    }

    for mode in modes:
        datasets = {
            split: MaestroTokenDataset(
                paths,
                tokenizer,
                data_config,
                mode=mode,
                training=split == "train",
            )
            for split, paths in split_paths.items()
        }
        metrics[mode] = train_one_model(
            mode,
            datasets,
            model_config,
            data_config,
            args.output_dir,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            device,
            args.seed,
        )
        if mode == "conditioned":
            write_examples(args.output_dir / "conditioned-best.pt", datasets["test"], tokenizer, args.output_dir)

    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Artifacts written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
