from __future__ import annotations

import json
from io import BytesIO

import pytest
import torch
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

from web.app import app, motif_from_request, parse_recorded_notes
from motifgen.v2 import CompleteNoteTokenizer, MotifContinuationTransformer, V2ModelConfig


def test_health_reports_missing_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.pt"))
    monkeypatch.delenv("MODEL_URL", raising=False)
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_ready"] is False


def test_home_page_contains_primary_studio():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Generate continuation" in response.text
        assert "Upload MIDI" in response.text
        assert 'id="category"' in response.text
        assert "Baroque / Classical" in response.text
        assert 'id="generation-progress"' in response.text
        assert 'id="result-texture"' in response.text
        assert 'id="motif-analysis"' in response.text
        assert 'id="piano-roll-stage"' in response.text
        assert 'id="download-image-button"' in response.text
        assert 'data-sample="two_hands"' in response.text


def test_analyze_recorded_motif_normalizes_and_reports_texture(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.pt"))
    monkeypatch.delenv("MODEL_URL", raising=False)
    motif = json.dumps(
        [
            {"pitch": 45, "start": 1.25, "end": 2.05, "velocity": 72},
            {"pitch": 52, "start": 1.25, "end": 2.05, "velocity": 75},
            {"pitch": 67, "start": 1.25, "end": 1.70, "velocity": 96},
            {"pitch": 72, "start": 1.70, "end": 2.10, "velocity": 101},
        ]
    )
    with TestClient(app) as client:
        response = client.post("/api/analyze", data={"motif_json": motif})

    assert response.status_code == 200
    payload = response.json()
    assert min(note["start"] for note in payload["notes"]) == 0
    assert payload["features"]["note_count"] == 4
    assert payload["features"]["pitch_min"] == 45
    assert payload["features"]["pitch_max"] == 72
    assert payload["features"]["bass_and_treble"] is True
    assert payload["features"]["texture"] == "full_polyphonic"


def test_analyze_rejects_missing_motif(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.pt"))
    monkeypatch.delenv("MODEL_URL", raising=False)
    with TestClient(app) as client:
        response = client.post("/api/analyze")
    assert response.status_code == 422
    assert "exactly one motif source" in response.json()["detail"]


def test_recorded_motif_validation():
    valid = json.dumps([
        {"pitch": 60, "start": 0, "end": 0.4, "velocity": 100},
        {"pitch": 64, "start": 0.5, "end": 0.9, "velocity": 90},
    ])
    assert len(parse_recorded_notes(valid)) == 2
    with pytest.raises(ValueError, match="between 2 and 500"):
        parse_recorded_notes("[]")


def test_generate_returns_503_without_model(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.pt"))
    monkeypatch.delenv("MODEL_URL", raising=False)
    with TestClient(app) as client:
        response = client.post(
            "/api/generate",
            data={"motif_json": "[]", "duration_seconds": "10", "temperature": "1.0"},
        )
        assert response.status_code == 503


@pytest.mark.asyncio
async def test_upload_validation_rejects_non_midi(tokenizer):
    upload = UploadFile(filename="motif.txt", file=BytesIO(b"not-midi"))
    with pytest.raises(ValueError, match=".mid or .midi"):
        await motif_from_request(tokenizer, None, upload)


def test_v2_api_applies_category_and_reports_inferred_texture(monkeypatch, tmp_path):
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
    checkpoint = tmp_path / "v2.pt"
    torch.save(
        {
            "format_version": 2,
            "model_kind": "motif_encoder_decoder_v2",
            "model_state": model.state_dict(),
            "model_config": config.to_dict(),
            "tokenizer_config": {"sample_rate": 100, "max_time_seconds": 30},
        },
        checkpoint,
    )
    monkeypatch.setenv("MODEL_PATH", str(checkpoint))
    monkeypatch.delenv("MODEL_URL", raising=False)
    motif = json.dumps(
        [
            {"pitch": 60, "start": 0, "end": 0.4, "velocity": 90},
            {"pitch": 62, "start": 0.5, "end": 0.9, "velocity": 94},
            {"pitch": 64, "start": 1.0, "end": 1.4, "velocity": 98},
        ]
    )
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["model_version"] == "v2"
        response = client.post(
            "/api/generate",
            data={
                "motif_json": motif,
                "duration_seconds": "5",
                "temperature": "0.8",
                "category": "romantic",
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["model_version"] == "v2"
        assert payload["category"] == "romantic"
        assert payload["category_applied"] is True
        assert payload["inferred_texture"] == "monophonic"
        assert payload["midi_base64"]

        invalid = client.post(
            "/api/generate",
            data={
                "motif_json": motif,
                "duration_seconds": "5",
                "temperature": "0.8",
                "category": "jazz",
            },
        )
        assert invalid.status_code == 422
