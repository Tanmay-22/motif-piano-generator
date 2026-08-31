from __future__ import annotations

import asyncio
import base64
import json
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TypeAlias

import torch

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from motifgen.generation import MotifGenerator
from motifgen.tokenizer import MidiTokenizer, RecordedNote
from motifgen.v2 import CompleteNoteTokenizer, MusicCategory, V2MotifGenerator
from motifgen.v2.features import extract_motif_features
from scripts.download_model import download_checkpoint


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "web" / "static"
DEFAULT_MODEL_PATH = ROOT / "artifacts" / "v2" / "conditioned-v2-best.pt"
MAX_UPLOAD_BYTES = 1_000_000
GENERATION_QUEUE_TIMEOUT_SECONDS = float(os.getenv("GENERATION_QUEUE_TIMEOUT_SECONDS", "5"))
GeneratorService: TypeAlias = MotifGenerator | V2MotifGenerator


def load_generator() -> tuple[GeneratorService | None, str | None]:
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    model_url = os.getenv("MODEL_URL")
    model_sha256 = os.getenv("MODEL_SHA256")
    try:
        if not model_path.exists() and model_url:
            download_checkpoint(model_url, model_path, model_sha256)
        if not model_path.exists():
            return None, f"Checkpoint not found at {model_path}"
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") == 2:
            return V2MotifGenerator.from_checkpoint(model_path), None
        return MotifGenerator.from_checkpoint(model_path), None
    except Exception as exc:
        return None, f"Checkpoint could not be loaded: {exc}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "1"))))
    app.state.generator, app.state.model_error = load_generator()
    app.state.generation_lock = asyncio.Lock()
    yield


app = FastAPI(
    title="Motif Piano Continuation API",
    description="Non-commercial motif-conditioned symbolic piano generation.",
    version="2.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
analysis_tokenizer = CompleteNoteTokenizer()


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def index() -> HTMLResponse:
    public_base_url = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html.replace("__PUBLIC_BASE_URL__", public_base_url))


@app.get("/health")
async def health(request: Request) -> dict[str, object]:
    ready = request.app.state.generator is not None
    generator = request.app.state.generator
    return {
        "status": "ok" if ready else "degraded",
        "model_ready": ready,
        "model_error": None if ready else request.app.state.model_error,
        "version": app.version,
        "model_version": (
            "v2" if isinstance(generator, V2MotifGenerator) else "v1" if generator else None
        ),
        "repository_url": os.getenv("GITHUB_REPOSITORY_URL"),
    }


def parse_recorded_notes(raw: str) -> list[RecordedNote]:
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Recorded motif data is not valid JSON.") from exc
    if not isinstance(values, list) or not 2 <= len(values) <= 500:
        raise ValueError("Record between 2 and 500 notes for a motif.")
    return [RecordedNote.from_mapping(value) for value in values]


def parse_editable_notes(raw: str) -> list[RecordedNote]:
    """Validate a complete edited result for temporary MIDI export."""

    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Edited MIDI data is not valid JSON.") from exc
    if not isinstance(values, list) or not 1 <= len(values) <= 2_000:
        raise ValueError("Edited MIDI must contain between 1 and 2,000 notes.")

    notes: list[RecordedNote] = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("Each edited note must be an object.")
        try:
            pitch = int(value["pitch"])
            start = float(value["start"])
            end = float(value["end"])
            velocity = int(value.get("velocity", 100))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Each edited note needs numeric pitch, start, end, and velocity values.") from exc
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("Edited note times must be finite numbers.")
        if not 21 <= pitch <= 108:
            raise ValueError("Piano pitches must be between MIDI 21 and 108.")
        if start < 0 or end <= start or end > 60:
            raise ValueError("Edited notes must use valid times between 0 and 60 seconds.")
        if not 1 <= velocity <= 127:
            raise ValueError("MIDI velocity must be between 1 and 127.")
        notes.append(RecordedNote(pitch=pitch, start=start, end=end, velocity=velocity))
    return sorted(notes, key=lambda note: (note.start, note.pitch, note.end))


async def motif_from_request(
    tokenizer: MidiTokenizer,
    motif_json: str | None,
    midi_file: UploadFile | None,
) -> list[int]:
    if bool(motif_json) == bool(midi_file):
        raise ValueError("Provide exactly one motif source: recorded notes or a MIDI file.")
    if motif_json:
        return tokenizer.notes_to_tokens(parse_recorded_notes(motif_json))

    payload = await read_midi_upload(midi_file)
    return tokenizer.midi_bytes_to_tokens(payload)


async def read_midi_upload(midi_file: UploadFile | None) -> bytes:
    assert midi_file is not None
    extension = Path(midi_file.filename or "").suffix.lower()
    if extension not in {".mid", ".midi"}:
        raise ValueError("Upload a .mid or .midi file.")
    payload = await midi_file.read(MAX_UPLOAD_BYTES + 1)
    await midi_file.close()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("MIDI uploads may be at most 1 MB.")
    if len(payload) < 14:
        raise ValueError("The uploaded MIDI file is empty or incomplete.")
    return payload


async def motif_notes_from_request(
    tokenizer: CompleteNoteTokenizer,
    motif_json: str | None,
    midi_file: UploadFile | None,
) -> list[RecordedNote]:
    if bool(motif_json) == bool(midi_file):
        raise ValueError("Provide exactly one motif source: recorded notes or a MIDI file.")
    if motif_json:
        return parse_recorded_notes(motif_json)
    return tokenizer.midi_bytes_to_notes(await read_midi_upload(midi_file))


@app.post("/api/analyze")
async def analyze_motif(
    motif_json: str | None = Form(default=None),
    midi_file: UploadFile | None = File(default=None),
) -> dict[str, object]:
    """Normalize and describe a motif without requiring a model checkpoint."""

    try:
        notes = await motif_notes_from_request(analysis_tokenizer, motif_json, midi_file)
        if len(notes) > 500:
            raise ValueError("Motifs may contain at most 500 notes.")
        features = extract_motif_features(notes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    first_onset = min(note.start for note in notes)
    normalized = [
        RecordedNote(
            pitch=note.pitch,
            start=max(0.0, note.start - first_onset),
            end=note.end - first_onset,
            velocity=note.velocity,
        )
        for note in notes
    ]
    return {
        "notes": [note.to_dict() for note in normalized],
        "features": {
            "texture": features.texture.value,
            "duration_seconds": features.duration_seconds,
            "note_count": features.note_count,
            "onset_count": features.onset_count,
            "note_density": features.note_density,
            "onset_density": features.onset_density,
            "average_polyphony": features.average_polyphony,
            "peak_polyphony": features.peak_polyphony,
            "average_chord_size": features.average_chord_size,
            "pitch_min": features.pitch_min,
            "pitch_max": features.pitch_max,
            "pitch_span": features.pitch_span,
            "velocity_mean": features.velocity_mean,
            "velocity_range": features.velocity_range,
            "median_onset_gap": features.median_onset_gap,
            "median_note_duration": features.median_note_duration,
            "bass_and_treble": features.bass_and_treble,
        },
    }


@app.post("/api/export-midi")
async def export_edited_midi(notes_json: str = Form(...)) -> dict[str, object]:
    """Create an ephemeral MIDI download from browser-edited note data."""

    try:
        notes = parse_editable_notes(notes_json)
        midi_bytes = analysis_tokenizer.notes_to_midi_bytes(notes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "midi_base64": base64.b64encode(midi_bytes).decode("ascii"),
        "note_count": len(notes),
        "duration_seconds": max(note.end for note in notes),
    }


@app.post("/api/generate")
async def generate(
    request: Request,
    motif_json: str | None = Form(default=None),
    midi_file: UploadFile | None = File(default=None),
    duration_seconds: int = Form(default=10),
    temperature: float = Form(default=1.0),
    category: str = Form(default="auto"),
) -> dict[str, object]:
    generator: GeneratorService | None = request.app.state.generator
    if generator is None:
        raise HTTPException(status_code=503, detail="The trained model is not configured on this service yet.")
    try:
        requested_category = MusicCategory.from_value(category)
        if isinstance(generator, V2MotifGenerator):
            motif_notes = await motif_notes_from_request(generator.tokenizer, motif_json, midi_file)
            maximum_notes = generator.model.config.max_motif_events - 2
            if len(motif_notes) > maximum_notes:
                raise ValueError(f"The v2 model accepts at most {maximum_notes} motif notes.")
        else:
            motif_tokens = await motif_from_request(generator.tokenizer, motif_json, midi_file)
            generator.tokenizer.prepare_motif(motif_tokens, generator.data_config.motif_max_tokens)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    lock: asyncio.Lock = request.app.state.generation_lock
    try:
        await asyncio.wait_for(lock.acquire(), timeout=GENERATION_QUEUE_TIMEOUT_SECONDS)
    except TimeoutError as exc:
        raise HTTPException(status_code=429, detail="Another composition is still running. Please try again shortly.") from exc

    try:
        try:
            if isinstance(generator, V2MotifGenerator):
                result = await asyncio.to_thread(
                    generator.generate,
                    motif_notes,
                    duration_seconds,
                    temperature,
                    requested_category,
                )
                midi_bytes = generator.tokenizer.notes_to_midi_bytes(result.all_notes)
                notes = [note.to_dict() for note in result.all_notes]
                total_duration = max((float(note["end"]) for note in notes), default=0.0)
                return {
                    "midi_base64": base64.b64encode(midi_bytes).decode("ascii"),
                    "notes": notes,
                    "duration_seconds": total_duration,
                    "continuation_duration_seconds": result.continuation_duration_seconds,
                    "motif_end_seconds": result.motif_end_seconds,
                    "reached_target_duration": result.reached_target_duration,
                    "timed_out": result.timed_out,
                    "model_version": "v2",
                    "category": result.category.value,
                    "category_applied": True,
                    "inferred_texture": result.motif_features.texture.value,
                }

            result = await asyncio.to_thread(
                generator.generate, motif_tokens, duration_seconds, temperature
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        midi_bytes = generator.tokenizer.tokens_to_midi_bytes(result.all_tokens)
        notes = [note.to_dict() for note in generator.tokenizer.tokens_to_notes(result.all_tokens)]
        motif_end = generator.tokenizer.elapsed_seconds(result.motif_tokens)
        total_duration = max((float(note["end"]) for note in notes), default=0.0)
        return {
            "midi_base64": base64.b64encode(midi_bytes).decode("ascii"),
            "notes": notes,
            "duration_seconds": total_duration,
            "continuation_duration_seconds": result.duration_seconds,
            "motif_end_seconds": motif_end,
            "reached_target_duration": result.reached_target_duration,
            "timed_out": False,
            "model_version": "v1",
            "category": requested_category.value,
            "category_applied": False,
            "inferred_texture": None,
        }
    finally:
        lock.release()
