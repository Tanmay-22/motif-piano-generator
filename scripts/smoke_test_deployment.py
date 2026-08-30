from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

import pretty_midi


SMOKE_MOTIF = [
    {"pitch": 60, "start": 0.0, "end": 0.38, "velocity": 82},
    {"pitch": 64, "start": 0.0, "end": 0.38, "velocity": 76},
    {"pitch": 67, "start": 0.0, "end": 0.38, "velocity": 78},
    {"pitch": 62, "start": 0.55, "end": 0.92, "velocity": 84},
    {"pitch": 65, "start": 0.55, "end": 0.92, "velocity": 79},
    {"pitch": 69, "start": 0.55, "end": 0.92, "velocity": 81},
]


def multipart_body(fields: dict[str, str]) -> tuple[bytes, str]:
    boundary = f"motif-{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.extend(
            (
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            )
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def fetch_json(request: urllib.request.Request, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def wait_for_v2(base_url: str, wake_timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + wake_timeout
    last_error = "service did not respond"
    while time.monotonic() < deadline:
        try:
            request = urllib.request.Request(f"{base_url}/health", headers={"Accept": "application/json"})
            health = fetch_json(request, timeout=min(30.0, wake_timeout))
            if not health.get("model_ready"):
                raise RuntimeError(health.get("model_error") or "model is not ready")
            if health.get("model_version") != "v2":
                raise RuntimeError(
                    f"expected model_version v2, received {health.get('model_version')!r}"
                )
            return health
        except (OSError, RuntimeError, ValueError) as exc:
            last_error = str(exc)
            print(f"Waiting for service: {last_error}")
            time.sleep(5)
    raise TimeoutError(f"Service did not become v2-ready within {wake_timeout:g}s: {last_error}")


def validate_generation(payload: dict[str, Any], expected_category: str) -> bytes:
    if payload.get("model_version") != "v2":
        raise ValueError("Generation response did not come from model v2.")
    if payload.get("category") != expected_category or payload.get("category_applied") is not True:
        raise ValueError("Generation response did not apply the requested category.")
    if payload.get("inferred_texture") not in {
        "monophonic",
        "light_polyphonic",
        "full_polyphonic",
    }:
        raise ValueError("Generation response is missing the inferred motif texture.")
    notes = payload.get("notes")
    if not isinstance(notes, list) or len(notes) <= len(SMOKE_MOTIF):
        raise ValueError("Generation response does not contain continuation notes.")
    try:
        midi_bytes = base64.b64decode(payload["midi_base64"], validate=True)
        midi = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    except Exception as exc:
        raise ValueError("Generation response does not contain a readable MIDI file.") from exc
    if not any(instrument.notes for instrument in midi.instruments):
        raise ValueError("Generated MIDI has no playable notes.")
    return midi_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wake and smoke-test the deployed Motif v2 service.")
    parser.add_argument("--base-url", required=True, help="For example https://motif-piano-generator.onrender.com")
    parser.add_argument("--wake-timeout", type=float, default=180)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument(
        "--category",
        choices=("auto", "baroque_classical", "romantic", "impressionist_modern"),
        default="romantic",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    health = wait_for_v2(base_url, args.wake_timeout)
    print(f"Ready: API {health.get('version')}, model {health.get('model_version')}")
    body, content_type = multipart_body(
        {
            "motif_json": json.dumps(SMOKE_MOTIF),
            "duration_seconds": "5",
            "temperature": "0.8",
            "category": args.category,
        }
    )
    request = urllib.request.Request(
        f"{base_url}/api/generate",
        data=body,
        method="POST",
        headers={"Content-Type": content_type, "Accept": "application/json"},
    )
    started = time.monotonic()
    payload = fetch_json(request, timeout=args.request_timeout)
    midi_bytes = validate_generation(payload, args.category)
    elapsed = time.monotonic() - started
    print(
        "Smoke test passed: "
        f"{len(payload['notes'])} notes, {len(midi_bytes)} MIDI bytes, "
        f"{elapsed:.1f}s, timed_out={payload.get('timed_out')}"
    )


if __name__ == "__main__":
    main()
