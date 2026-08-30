from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


MAESTRO_MIDI_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"


def download_maestro(destination: Path) -> Path:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "maestro-v3.0.0-midi.zip"
    extracted = destination / "maestro-v3.0.0"
    metadata = next(extracted.rglob("maestro-v3.0.0.csv"), None) if extracted.exists() else None
    if metadata:
        print(f"MAESTRO is already available at {extracted}")
        return extracted

    if not archive.exists():
        print(f"Downloading MAESTRO v3 MIDI archive to {archive}")
        with urllib.request.urlopen(MAESTRO_MIDI_URL) as source, archive.open("wb") as target:
            shutil.copyfileobj(source, target)

    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (extracted / member.filename).resolve()
            if extracted not in target.parents and target != extracted:
                raise ValueError("Dataset archive contains an unsafe path.")
        bundle.extractall(extracted)
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the MAESTRO v3 MIDI dataset.")
    parser.add_argument("--destination", type=Path, default=Path("data"))
    args = parser.parse_args()
    download_maestro(args.destination)


if __name__ == "__main__":
    main()
