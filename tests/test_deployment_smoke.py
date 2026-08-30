from __future__ import annotations

import base64
import io

import pretty_midi
import pytest

from scripts.smoke_test_deployment import SMOKE_MOTIF, multipart_body, validate_generation


def _midi_base64() -> str:
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    piano.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0, end=0.5))
    midi.instruments.append(piano)
    destination = io.BytesIO()
    midi.write(destination)
    return base64.b64encode(destination.getvalue()).decode("ascii")


def test_multipart_body_contains_every_field():
    body, content_type = multipart_body({"category": "romantic", "duration_seconds": "5"})
    assert content_type.startswith("multipart/form-data; boundary=motif-")
    assert b'name="category"' in body
    assert b"romantic" in body
    assert body.endswith(b"--\r\n")


def test_deployment_payload_validation_accepts_v2_midi():
    payload = {
        "model_version": "v2",
        "category": "romantic",
        "category_applied": True,
        "inferred_texture": "full_polyphonic",
        "notes": [*SMOKE_MOTIF, {"pitch": 71, "start": 1.0, "end": 1.4, "velocity": 85}],
        "midi_base64": _midi_base64(),
    }
    assert validate_generation(payload, "romantic").startswith(b"MThd")


def test_deployment_payload_validation_rejects_v1():
    with pytest.raises(ValueError, match="model v2"):
        validate_generation({"model_version": "v1"}, "romantic")
