from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from motifgen.v2 import (
    CompleteNoteTokenizer,
    MotifContinuationTransformer,
    V2ModelConfig,
    factorized_event_loss,
)
from training.download_data import download_maestro
from training.v2_batching import V2Batch, collate_v2_phrase_pairs
from training.v2_dataset import (
    MaestroV2PhraseDataset,
    PhraseExtractionConfig,
    load_v2_split_records,
)


CHECKPOINT_FORMAT_VERSION = 2


@dataclass
class TrainingProgress:
    global_step: int = 0
    epoch: int = 0
    next_batch_index: int = 0
    best_validation_loss: float = float("inf")
    evaluations_without_improvement: int = 0
    history: list[dict[str, float | int]] | None = None

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = []


class LossAccumulator:
    def __init__(self) -> None:
        self.event_total = 0.0
        self.note_totals = {
            name: 0.0
            for name in (
                "delta_coarse",
                "delta_fine",
                "pitch",
                "duration_coarse",
                "duration_fine",
                "velocity",
            )
        }
        self.valid_events = 0
        self.note_events = 0

    def update(self, loss) -> None:
        self.event_total += float(loss.event_type.item()) * loss.valid_events
        self.valid_events += loss.valid_events
        for name in self.note_totals:
            self.note_totals[name] += float(getattr(loss, name).item()) * loss.note_events
        self.note_events += loss.note_events

    def metrics(self) -> dict[str, float | int]:
        event_loss = self.event_total / max(self.valid_events, 1)
        note_losses = {
            name: value / max(self.note_events, 1) for name, value in self.note_totals.items()
        }
        total = event_loss + (sum(note_losses.values()) / len(note_losses))
        return {
            "loss": total,
            "event_type_loss": event_loss,
            **{f"{name}_loss": value for name, value in note_losses.items()},
            "valid_events": self.valid_events,
            "note_events": self.note_events,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def learning_rate_multiplier(step: int, warmup_steps: int, max_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max((step + 1) / warmup_steps, 1e-3)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    progress = max(0.0, min(1.0, progress))
    return 0.1 + (0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def make_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, max_steps: int
) -> LambdaLR:
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: learning_rate_multiplier(step, warmup_steps, max_steps),
    )


def atomic_torch_save(value: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(destination)


def atomic_json_save(value: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(destination)


def inference_checkpoint(
    model: MotifContinuationTransformer,
    tokenizer: CompleteNoteTokenizer,
    phrase_config: PhraseExtractionConfig,
    progress: TrainingProgress,
    validation_metrics: dict[str, float | int],
) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_kind": "motif_encoder_decoder_v2",
        "model_state": model.state_dict(),
        "model_config": model.config.to_dict(),
        "tokenizer_config": {
            "sample_rate": tokenizer.sample_rate,
            "max_time_seconds": tokenizer.max_time_seconds,
        },
        "phrase_config": asdict(phrase_config),
        "global_step": progress.global_step,
        "validation_metrics": validation_metrics,
        "categories": [
            "auto",
            "baroque_classical",
            "romantic",
            "impressionist_modern",
        ],
    }


def training_checkpoint(
    model: MotifContinuationTransformer,
    tokenizer: CompleteNoteTokenizer,
    phrase_config: PhraseExtractionConfig,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    progress: TrainingProgress,
) -> dict[str, Any]:
    return {
        **inference_checkpoint(model, tokenizer, phrase_config, progress, {}),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "progress": asdict(progress),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_training_checkpoint(
    checkpoint: dict[str, Any],
    model: MotifContinuationTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
) -> TrainingProgress:
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("This is not a compatible v2 training checkpoint.")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    scaler.load_state_dict(checkpoint.get("scaler_state", {}))
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state"):
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    return TrainingProgress(**checkpoint["progress"])


def shuffled_motif_batch(batch: V2Batch) -> V2Batch:
    if batch.motif_events.size(0) < 2:
        return batch
    permutation = torch.roll(torch.arange(batch.motif_events.size(0), device=batch.motif_events.device), 1)
    return V2Batch(
        motif_events=batch.motif_events[permutation],
        motif_padding_mask=batch.motif_padding_mask[permutation],
        decoder_inputs=batch.decoder_inputs,
        decoder_targets=batch.decoder_targets,
        decoder_padding_mask=batch.decoder_padding_mask,
        # Keep the requested period fixed, changing only motif-derived context.
        category_ids=batch.category_ids,
        texture_ids=batch.texture_ids[permutation],
        motif_controls=batch.motif_controls[permutation],
    )


@torch.inference_mode()
def evaluate_v2(
    model: MotifContinuationTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    compare_shuffled_motif: bool = True,
    max_batches: int | None = None,
) -> dict[str, float | int]:
    model.eval()
    correct = LossAccumulator()
    shuffled = LossAccumulator()
    for batch_index, cpu_batch in enumerate(tqdm(loader, desc="Evaluating v2", leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = cpu_batch.to(device)
        logits = model(
            batch.motif_events,
            batch.decoder_inputs,
            batch.category_ids,
            batch.texture_ids,
            batch.motif_controls,
            batch.motif_padding_mask,
            batch.decoder_padding_mask,
        )
        correct.update(factorized_event_loss(logits, batch.decoder_targets))
        if compare_shuffled_motif:
            changed = shuffled_motif_batch(batch)
            changed_logits = model(
                changed.motif_events,
                changed.decoder_inputs,
                changed.category_ids,
                changed.texture_ids,
                changed.motif_controls,
                changed.motif_padding_mask,
                changed.decoder_padding_mask,
            )
            shuffled.update(factorized_event_loss(changed_logits, changed.decoder_targets))

    metrics = correct.metrics()
    if compare_shuffled_motif:
        shuffled_metrics = shuffled.metrics()
        metrics["shuffled_motif_loss"] = float(shuffled_metrics["loss"])
        metrics["motif_dependency_gap"] = float(shuffled_metrics["loss"]) - float(metrics["loss"])
    return metrics


def make_training_loader(
    dataset: MaestroV2PhraseDataset,
    batch_size: int,
    epoch: int,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    category_counts = Counter(record.category for record, _ in dataset.pieces)
    weights = [
        1.0 / category_counts[dataset.pieces[index % len(dataset.pieces)][0].category]
        for index in range(len(dataset))
    ]
    generator = torch.Generator().manual_seed(seed + epoch)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_v2_phrase_pairs,
    )


def make_evaluation_loader(
    dataset: MaestroV2PhraseDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_v2_phrase_pairs,
    )


def evaluation_batch_limit(value: int) -> int | None:
    """Interpret zero as a full-split evaluation and reject negative limits."""

    if value < 0:
        raise ValueError("evaluation-max-batches cannot be negative.")
    return None if value == 0 else value


def resolve_resume_path(value: str, output_dir: Path) -> Path | None:
    if value.lower() == "none":
        return None
    if value.lower() == "auto":
        candidate = output_dir / "latest.pt"
        return candidate if candidate.exists() else None
    candidate = Path(value)
    if not candidate.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {candidate}")
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the category-aware v2 motif continuation model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v2"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/v2-cache"))
    parser.add_argument("--resume", default="auto", help="auto, none, or a trusted checkpoint path")
    parser.add_argument("--evaluate-only", type=Path)
    parser.add_argument("--max-steps", type=int, default=30_000)
    parser.add_argument("--session-steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--evaluate-every", type=int, default=1_000)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--train-examples-per-piece", type=int, default=8)
    parser.add_argument("--evaluation-examples-per-piece", type=int, default=1)
    parser.add_argument("--style-dropout", type=float, default=0.25)
    parser.add_argument("--evaluation-max-batches", type=int, default=50)
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    positive = (
        "max_steps",
        "session_steps",
        "batch_size",
        "gradient_accumulation",
        "save_every",
        "evaluate_every",
        "train_examples_per_piece",
        "evaluation_examples_per_piece",
    )
    if any(getattr(args, name) < 1 for name in positive):
        parser.error("Step, batch, accumulation, save, evaluation, and example counts must be positive.")
    if args.evaluation_max_batches < 0:
        parser.error("evaluation-max-batches must be zero (all batches) or a positive integer.")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_limit = evaluation_batch_limit(args.evaluation_max_batches)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    print(f"Training v2 on {device}; AMP={amp_enabled}")

    checkpoint_path = args.evaluate_only or resolve_resume_path(args.resume, args.output_dir)
    loaded_checkpoint = (
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint_path is not None
        else None
    )
    if loaded_checkpoint is not None:
        tokenizer = CompleteNoteTokenizer(**loaded_checkpoint["tokenizer_config"])
        phrase_config = PhraseExtractionConfig(**loaded_checkpoint["phrase_config"])
        model_config = V2ModelConfig.from_dict(loaded_checkpoint["model_config"])
    else:
        tokenizer = CompleteNoteTokenizer()
        phrase_config = PhraseExtractionConfig()
        model_config = V2ModelConfig.from_tokenizer(tokenizer)

    data_root = download_maestro(args.data_dir)
    records = {
        split: load_v2_split_records(data_root, split, args.limit_per_split)
        for split in ("train", "validation", "test")
    }
    datasets = {
        split: MaestroV2PhraseDataset(
            split_records,
            tokenizer,
            phrase_config,
            training=split == "train",
            examples_per_piece=(
                args.train_examples_per_piece
                if split == "train"
                else args.evaluation_examples_per_piece
            ),
            style_dropout_probability=args.style_dropout,
            seed=args.seed,
            cache_path=args.cache_dir / f"{split}-notes.npz",
        )
        for split, split_records in records.items()
    }
    pin_memory = device.type == "cuda"
    validation_loader = make_evaluation_loader(
        datasets["validation"], args.batch_size, args.num_workers, pin_memory
    )
    test_loader = make_evaluation_loader(
        datasets["test"], args.batch_size, args.num_workers, pin_memory
    )

    model = MotifContinuationTransformer(model_config).to(device)
    if args.evaluate_only:
        model.load_state_dict(loaded_checkpoint["model_state"])
        validation_metrics = evaluate_v2(
            model,
            validation_loader,
            device,
            max_batches=evaluation_limit,
        )
        test_metrics = evaluate_v2(
            model,
            test_loader,
            device,
            max_batches=evaluation_limit,
        )
        cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(args.seed)
            untrained_model = MotifContinuationTransformer(model_config).to(device)
        untrained_validation = evaluate_v2(
            untrained_model,
            validation_loader,
            device,
            compare_shuffled_motif=False,
            max_batches=evaluation_limit,
        )
        validation_improvement = float(untrained_validation["loss"]) - float(
            validation_metrics["loss"]
        )
        report = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "checkpoint": args.evaluate_only.name,
            "global_step": int(loaded_checkpoint.get("global_step", 0)),
            "evaluation_scope": "full_split" if evaluation_limit is None else f"first_{evaluation_limit}_batches",
            "validation": validation_metrics,
            "test": test_metrics,
            "untrained_validation": untrained_validation,
            "validation_improvement_over_untrained": validation_improvement,
            "quality_gates": {
                "validation_beats_untrained": validation_improvement > 0,
                "validation_motif_dependency_positive": float(
                    validation_metrics.get("motif_dependency_gap", 0.0)
                ) > 0,
                "test_motif_dependency_positive": float(
                    test_metrics.get("motif_dependency_gap", 0.0)
                ) > 0,
            },
        }
        atomic_json_save(report, args.output_dir / "evaluation-v2.json")
        print(json.dumps(report, indent=2))
        return

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = make_scheduler(optimizer, args.warmup_steps, args.max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    progress = TrainingProgress()
    if loaded_checkpoint is not None:
        progress = restore_training_checkpoint(
            loaded_checkpoint, model, optimizer, scheduler, scaler
        )
        print(f"Resumed at optimizer step {progress.global_step:,}, epoch {progress.epoch}.")

    session_end_step = min(args.max_steps, progress.global_step + args.session_steps)
    latest_path = args.output_dir / "latest.pt"
    best_path = args.output_dir / "conditioned-v2-best.pt"
    stop_training = False

    while progress.global_step < session_end_step and not stop_training:
        datasets["train"].set_epoch(progress.epoch)
        train_loader = make_training_loader(
            datasets["train"],
            args.batch_size,
            progress.epoch,
            args.seed,
            args.num_workers,
            pin_memory,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress_bar = tqdm(train_loader, desc=f"v2 epoch {progress.epoch}")
        for batch_index, cpu_batch in enumerate(progress_bar):
            if batch_index < progress.next_batch_index:
                continue
            batch = cpu_batch.to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(
                    batch.motif_events,
                    batch.decoder_inputs,
                    batch.category_ids,
                    batch.texture_ids,
                    batch.motif_controls,
                    batch.motif_padding_mask,
                    batch.decoder_padding_mask,
                )
                loss = factorized_event_loss(logits, batch.decoder_targets)
                scaled_loss = loss.total / args.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            should_step = (
                (batch_index + 1) % args.gradient_accumulation == 0
                or batch_index + 1 == len(train_loader)
            )
            progress.next_batch_index = batch_index + 1
            if not should_step:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            progress.global_step += 1
            progress_bar.set_postfix(
                step=progress.global_step,
                loss=f"{loss.total.item():.3f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

            if progress.global_step % args.save_every == 0:
                atomic_torch_save(
                    training_checkpoint(
                        model, tokenizer, phrase_config, optimizer, scheduler, scaler, progress
                    ),
                    latest_path,
                )

            if progress.global_step % args.evaluate_every == 0:
                validation = evaluate_v2(
                    model,
                    validation_loader,
                    device,
                    max_batches=evaluation_limit,
                )
                validation_loss = float(validation["loss"])
                entry: dict[str, float | int] = {
                    "step": progress.global_step,
                    **validation,
                }
                progress.history.append(entry)
                improved = validation_loss < progress.best_validation_loss
                if improved:
                    progress.best_validation_loss = validation_loss
                    progress.evaluations_without_improvement = 0
                    atomic_torch_save(
                        inference_checkpoint(
                            model, tokenizer, phrase_config, progress, validation
                        ),
                        best_path,
                    )
                else:
                    progress.evaluations_without_improvement += 1
                atomic_json_save(
                    {
                        "model_config": model_config.to_dict(),
                        "phrase_config": asdict(phrase_config),
                        "progress": asdict(progress),
                    },
                    args.output_dir / "metrics-v2.json",
                )
                model.train()
                if progress.evaluations_without_improvement >= args.early_stopping_patience:
                    print("Early stopping: validation loss has stopped improving.")
                    stop_training = True
                    break

            if progress.global_step >= session_end_step:
                break

        if progress.next_batch_index >= len(train_loader):
            progress.epoch += 1
            progress.next_batch_index = 0

    atomic_torch_save(
        training_checkpoint(model, tokenizer, phrase_config, optimizer, scheduler, scaler, progress),
        latest_path,
    )
    if progress.global_step >= args.max_steps or stop_training:
        if not best_path.exists():
            validation = evaluate_v2(
                model,
                validation_loader,
                device,
                max_batches=evaluation_limit,
            )
            atomic_torch_save(
                inference_checkpoint(model, tokenizer, phrase_config, progress, validation),
                best_path,
            )
        best = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(best["model_state"])
        test_metrics = evaluate_v2(
            model,
            test_loader,
            device,
            max_batches=evaluation_limit,
        )
        atomic_json_save(
            {
                "best_validation": best["validation_metrics"],
                "test": test_metrics,
                "global_step": progress.global_step,
            },
            args.output_dir / "final-report-v2.json",
        )
        print(json.dumps({"test": test_metrics}, indent=2))
    else:
        print(
            f"Session finished at step {progress.global_step:,}. "
            "Rerun the same command to resume from latest.pt."
        )


if __name__ == "__main__":
    main()
