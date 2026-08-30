from __future__ import annotations

import hashlib
from email.message import Message
from io import BytesIO

import pytest

from scripts.download_model import download_checkpoint


class FakeResponse(BytesIO):
    def __init__(self, payload: bytes, declared_length: int | None = None) -> None:
        super().__init__(payload)
        self.headers = Message()
        if declared_length is not None:
            self.headers["Content-Length"] = str(declared_length)


def test_checkpoint_download_streams_and_verifies(monkeypatch, tmp_path):
    payload = b"trained-v2-checkpoint"
    expected = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        "scripts.download_model.urllib.request.urlopen",
        lambda *_args, **_kwargs: FakeResponse(payload, len(payload)),
    )
    destination = tmp_path / "model.pt"
    assert download_checkpoint(
        "https://example.test/model.pt",
        destination,
        expected,
        require_sha256=True,
    ) == destination.resolve()
    assert destination.read_bytes() == payload


def test_checkpoint_download_rejects_bad_configuration_before_network(tmp_path):
    with pytest.raises(ValueError, match="HTTPS"):
        download_checkpoint("http://example.test/model.pt", tmp_path / "model.pt")
    with pytest.raises(ValueError, match="required"):
        download_checkpoint(
            "https://example.test/model.pt",
            tmp_path / "model.pt",
            require_sha256=True,
        )
    with pytest.raises(ValueError, match="64 hexadecimal"):
        download_checkpoint(
            "https://example.test/model.pt",
            tmp_path / "model.pt",
            "not-a-digest",
        )


def test_checkpoint_download_rejects_oversize_and_cleans_partial(monkeypatch, tmp_path):
    payload = b"too large"
    monkeypatch.setattr(
        "scripts.download_model.urllib.request.urlopen",
        lambda *_args, **_kwargs: FakeResponse(payload, len(payload)),
    )
    destination = tmp_path / "model.pt"
    with pytest.raises(ValueError, match="size limit"):
        download_checkpoint(
            "https://example.test/model.pt",
            destination,
            max_bytes=4,
        )
    assert not destination.exists()
    assert not destination.with_suffix(".pt.part").exists()
