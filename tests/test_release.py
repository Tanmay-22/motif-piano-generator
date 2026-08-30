from __future__ import annotations

import json
import sys

import torch

from motifgen.tokenizer import RecordedNote
from motifgen.v2 import CompleteNoteTokenizer, MotifContinuationTransformer, V2ModelConfig
from scripts.prepare_v2_release import (
    main as prepare_release,
    release_asset_url,
    release_notes_markdown,
    release_quality_failures,
    render_environment_text,
    sha256_file,
)


def _passing_report() -> dict:
    metrics = {"loss": 2.0, "note_events": 100, "motif_dependency_gap": 0.1}
    return {
        "evaluation_scope": "full_split",
        "validation": dict(metrics),
        "test": dict(metrics),
        "quality_gates": {
            "validation_beats_untrained": True,
            "validation_motif_dependency_positive": True,
            "test_motif_dependency_positive": True,
        },
    }


def test_release_quality_report_accepts_full_passing_evaluation():
    assert release_quality_failures(_passing_report()) == []


def test_release_quality_report_rejects_partial_or_failed_evaluation():
    report = _passing_report()
    report["evaluation_scope"] = "first_50_batches"
    report["quality_gates"]["test_motif_dependency_positive"] = False
    failures = release_quality_failures(report)
    assert "evaluation_scope must be full_split" in failures
    assert "quality gate failed: test_motif_dependency_positive" in failures


def test_release_sha256_is_streamed_and_stable(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"motif-v2")
    assert sha256_file(artifact) == "2f5db53ecb3908adc8bee93f2036bfff4094bf81978ec30546e96b646a6f1df7"


def test_release_urls_and_render_environment_are_pinned():
    expected_url = (
        "https://github.com/Tanmay-22/motif-piano-generator/releases/download/"
        "model-v2.0.0/conditioned-v2-best.pt"
    )
    assert (
        release_asset_url(
            "Tanmay-22/motif-piano-generator",
            "model-v2.0.0",
            "conditioned-v2-best.pt",
        )
        == expected_url
    )
    rendered = render_environment_text(
        "Tanmay-22/motif-piano-generator",
        "model-v2.0.0",
        "a" * 64,
        "https://motif-piano-generator.onrender.com/",
    )
    assert f"MODEL_URL={expected_url}" in rendered
    assert f"MODEL_SHA256={'a' * 64}" in rendered
    assert "PUBLIC_BASE_URL=https://motif-piano-generator.onrender.com\n" in rendered


def test_release_notes_are_filled_from_measured_results():
    report = _passing_report()
    report["validation_improvement_over_untrained"] = 1.25
    notes = release_notes_markdown("model-v2.0.0", report, {"global_step": 12_000})
    assert "Held-out test loss: `2.000000`" in notes
    assert "Improvement over an untrained model: `1.250000`" in notes
    assert "Training optimizer step: `12000`" in notes


def test_release_packager_builds_verified_bundle(monkeypatch, tmp_path):
    tokenizer = CompleteNoteTokenizer()
    config = V2ModelConfig.from_tokenizer(
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
    model = MotifContinuationTransformer(config)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "format_version": 2,
            "model_kind": "motif_encoder_decoder_v2",
            "model_state": model.state_dict(),
            "model_config": config.to_dict(),
            "tokenizer_config": {"sample_rate": 100, "max_time_seconds": 30},
            "global_step": 12000,
            "categories": [
                "auto",
                "baroque_classical",
                "romantic",
                "impressionist_modern",
            ],
        },
        checkpoint_path,
    )
    report = _passing_report()
    report["validation_improvement_over_untrained"] = 1.25
    evaluation_path = tmp_path / "evaluation-v2.json"
    evaluation_path.write_text(json.dumps(report), encoding="utf-8")
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "examples-v2.json").write_text(
        json.dumps({"examples": [{"number": 1}]}), encoding="utf-8"
    )
    (examples / "example-1-generated.mid").write_bytes(
        tokenizer.notes_to_midi_bytes([RecordedNote(60, 0, 0.5, 90)])
    )
    destination = tmp_path / "release"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_v2_release",
            "--checkpoint",
            str(checkpoint_path),
            "--evaluation",
            str(evaluation_path),
            "--examples-dir",
            str(examples),
            "--output-dir",
            str(destination),
        ],
    )
    prepare_release()
    manifest = json.loads((destination / "release-manifest-v2.json").read_text())
    assert manifest["checkpoint"]["format_version"] == 2
    assert manifest["example_count"] == 1
    assert (destination / "RELEASE_NOTES.md").exists()
    assert (destination / "render-env-v2.txt").exists()
    assert manifest["checkpoint"]["download_url"].endswith(
        "/model-v2.0.0/conditioned-v2-best.pt"
    )
    assert (destination / "SHA256SUMS.txt").exists()
    assert destination.with_suffix(".zip").exists()
