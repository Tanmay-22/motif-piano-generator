from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DEFAULT_MAX_CHECKPOINT_BYTES = 500 * 1024 * 1024


def _validated_sha256(value: str | None, *, required: bool) -> str | None:
    normalized = value.strip().lower() if value else None
    if required and not normalized:
        raise ValueError("MODEL_SHA256 is required for this checkpoint download.")
    if normalized and (
        len(normalized) != 64
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise ValueError("MODEL_SHA256 must contain exactly 64 hexadecimal characters.")
    return normalized


def download_checkpoint(
    url: str,
    destination: Path,
    expected_sha256: str | None = None,
    *,
    require_sha256: bool = False,
    max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
) -> Path:
    if urlparse(url).scheme.lower() != "https":
        raise ValueError("Checkpoint URLs must use HTTPS.")
    expected_digest = _validated_sha256(expected_sha256, required=require_sha256)
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive.")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        digest = hashlib.sha256()
        downloaded_bytes = 0
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as target:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                raise ValueError("Checkpoint download exceeds the configured size limit.")
            while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                downloaded_bytes += len(chunk)
                if downloaded_bytes > max_bytes:
                    raise ValueError("Checkpoint download exceeds the configured size limit.")
                digest.update(chunk)
                target.write(chunk)
        if downloaded_bytes == 0:
            raise ValueError("Checkpoint download was empty.")
        if expected_digest:
            if digest.hexdigest().lower() != expected_digest:
                raise ValueError("Downloaded checkpoint SHA-256 does not match MODEL_SHA256.")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the pinned public model checkpoint.")
    parser.add_argument("--url", default=os.getenv("MODEL_URL"))
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(os.getenv("MODEL_PATH", "artifacts/v2/conditioned-v2-best.pt")),
    )
    parser.add_argument("--sha256", default=os.getenv("MODEL_SHA256"))
    parser.add_argument(
        "--require-sha256",
        action="store_true",
        help="Fail before downloading unless a valid SHA-256 value is supplied.",
    )
    args = parser.parse_args()
    if not args.url:
        parser.error("Provide --url or set MODEL_URL")
    print(
        download_checkpoint(
            args.url,
            args.destination,
            args.sha256,
            require_sha256=args.require_sha256,
        )
    )


if __name__ == "__main__":
    main()
