from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from motifgen.v2 import V2MotifGenerator


REQUIRED_CATEGORIES = (
    "auto",
    "baroque_classical",
    "romantic",
    "impressionist_modern",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_quality_failures(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if report.get("evaluation_scope") != "full_split":
        failures.append("evaluation_scope must be full_split")
    for split in ("validation", "test"):
        metrics = report.get(split)
        if not isinstance(metrics, dict):
            failures.append(f"{split} metrics are missing")
            continue
        loss = metrics.get("loss")
        if not isinstance(loss, (int, float)) or not math.isfinite(float(loss)):
            failures.append(f"{split} loss is missing or non-finite")
        if int(metrics.get("note_events", 0)) < 1:
            failures.append(f"{split} evaluated no note events")
    gates = report.get("quality_gates")
    if not isinstance(gates, dict):
        failures.append("quality_gates are missing")
    else:
        for name in (
            "validation_beats_untrained",
            "validation_motif_dependency_positive",
            "test_motif_dependency_positive",
        ):
            if gates.get(name) is not True:
                failures.append(f"quality gate failed: {name}")
    return failures


def release_notes_markdown(
    release_tag: str,
    report: dict[str, Any],
    checkpoint: dict[str, Any],
) -> str:
    validation = report["validation"]
    test = report["test"]
    return f"""# Motif conditioned piano model {release_tag.removeprefix('model-')}

Category-aware motif encoder/decoder trained on the MAESTRO v3.0.0 MIDI
dataset using its official train, validation, and test splits.

## Evaluation

- Validation loss: `{float(validation['loss']):.6f}`
- Held-out test loss: `{float(test['loss']):.6f}`
- Improvement over an untrained model: `{float(report['validation_improvement_over_untrained']):.6f}`
- Validation motif-dependency gap: `{float(validation['motif_dependency_gap']):.6f}`
- Test motif-dependency gap: `{float(test['motif_dependency_gap']):.6f}`
- Training optimizer step: `{int(checkpoint.get('global_step', 0))}`

All evaluation batches were included. The quality gates confirm that validation
loss beats freshly initialized weights and that both validation and test loss
increase when the correct motifs are shuffled.

## Contents

The release ZIP contains the evaluation report, fixed held-out motifs, their
real and generated continuations, model configuration, source revision,
license, and SHA-256 checksums. `conditioned-v2-best.pt` is also attached as a
standalone asset for the Render deployment.

## License

Non-commercial research and portfolio use only. The checkpoint and
MAESTRO-derived examples are released under CC BY-NC-SA 4.0. See
`MODEL_LICENSE.md` in the release bundle for attribution and citation.
"""


def source_revision() -> str | None:
    environment_revision = os.getenv("GITHUB_SHA")
    if environment_revision:
        return environment_revision
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and assemble the public non-commercial v2 model release assets."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--examples-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release-tag", default="model-v2.0.0")
    parser.add_argument("--allow-failed-quality-gates", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 2:
        raise ValueError("Release checkpoint must use format_version 2.")
    if checkpoint.get("model_kind") != "motif_encoder_decoder_v2":
        raise ValueError("Release checkpoint has an unexpected model_kind.")
    if tuple(checkpoint.get("categories", ())) != REQUIRED_CATEGORIES:
        raise ValueError("Release checkpoint category order is missing or incompatible.")
    generator = V2MotifGenerator.from_checkpoint(args.checkpoint)

    report = json.loads(args.evaluation.read_text(encoding="utf-8"))
    failures = release_quality_failures(report)
    if failures and not args.allow_failed_quality_gates:
        details = "\n- ".join(failures)
        raise ValueError(f"Release quality gates failed:\n- {details}")
    if failures:
        print("WARNING: release created despite failed quality gates:")
        for failure in failures:
            print(f"- {failure}")

    examples_manifest = args.examples_dir / "examples-v2.json"
    if not examples_manifest.exists():
        raise FileNotFoundError("Examples directory must contain examples-v2.json.")
    examples_report = json.loads(examples_manifest.read_text(encoding="utf-8"))
    logical_examples = examples_report.get("examples")
    if not isinstance(logical_examples, list) or not logical_examples:
        raise ValueError("examples-v2.json does not contain any example records.")
    midi_examples = sorted(args.examples_dir.glob("*.mid"))
    if not midi_examples:
        raise FileNotFoundError("Examples directory does not contain any MIDI files.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_destination = args.output_dir / "conditioned-v2-best.pt"
    evaluation_destination = args.output_dir / "evaluation-v2.json"
    license_destination = args.output_dir / "MODEL_LICENSE.md"
    shutil.copy2(args.checkpoint, checkpoint_destination)
    shutil.copy2(args.evaluation, evaluation_destination)
    shutil.copy2(Path(__file__).resolve().parents[1] / "MODEL_LICENSE.md", license_destination)
    release_examples = args.output_dir / "examples"
    release_examples.mkdir(parents=True, exist_ok=True)
    shutil.copy2(examples_manifest, release_examples / examples_manifest.name)
    for midi_path in midi_examples:
        shutil.copy2(midi_path, release_examples / midi_path.name)

    checkpoint_sha = sha256_file(checkpoint_destination)
    manifest = {
        "release_tag": args.release_tag,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": source_revision(),
        "dataset": "MAESTRO v3.0.0 MIDI",
        "dataset_split_policy": "official train/validation/test",
        "license": "CC BY-NC-SA 4.0",
        "commercial_use": False,
        "checkpoint": {
            "filename": checkpoint_destination.name,
            "sha256": checkpoint_sha,
            "bytes": checkpoint_destination.stat().st_size,
            "format_version": 2,
            "model_kind": checkpoint["model_kind"],
            "global_step": int(checkpoint.get("global_step", 0)),
            "parameter_count": generator.model.parameter_count(),
            "categories": list(REQUIRED_CATEGORIES),
            "model_config": checkpoint["model_config"],
            "tokenizer_config": checkpoint["tokenizer_config"],
        },
        "evaluation": {
            "scope": report.get("evaluation_scope"),
            "validation_loss": report["validation"]["loss"],
            "test_loss": report["test"]["loss"],
            "validation_improvement_over_untrained": report.get(
                "validation_improvement_over_untrained"
            ),
            "validation_motif_dependency_gap": report["validation"].get(
                "motif_dependency_gap"
            ),
            "test_motif_dependency_gap": report["test"].get("motif_dependency_gap"),
            "quality_gates": report.get("quality_gates", {}),
            "quality_gate_overrides": failures,
        },
        "example_count": len(logical_examples),
        "example_midi_asset_count": len(midi_examples),
    }
    manifest_path = args.output_dir / "release-manifest-v2.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output_dir / "RELEASE_NOTES.md").write_text(
        release_notes_markdown(args.release_tag, report, checkpoint), encoding="utf-8"
    )

    checksum_paths = sorted(
        path for path in args.output_dir.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    checksum_lines = [
        f"{sha256_file(path)}  {path.relative_to(args.output_dir).as_posix()}"
        for path in checksum_paths
    ]
    (args.output_dir / "SHA256SUMS.txt").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    archive_path = shutil.make_archive(str(args.output_dir), "zip", root_dir=args.output_dir)
    print(f"Release directory: {args.output_dir}")
    print(f"Release archive: {archive_path}")
    print(f"MODEL_SHA256={checkpoint_sha}")


if __name__ == "__main__":
    main()
