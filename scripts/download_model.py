from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import urllib.request
from pathlib import Path


def download_checkpoint(url: str, destination: Path, expected_sha256: str | None = None) -> Path:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as target:
            shutil.copyfileobj(response, target)
        if expected_sha256:
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            if digest.lower() != expected_sha256.lower():
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
    args = parser.parse_args()
    if not args.url:
        parser.error("Provide --url or set MODEL_URL")
    print(download_checkpoint(args.url, args.destination, args.sha256))


if __name__ == "__main__":
    main()
