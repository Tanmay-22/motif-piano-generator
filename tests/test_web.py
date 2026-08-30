from __future__ import annotations

import json
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

from web.app import app, motif_from_request, parse_recorded_notes


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
