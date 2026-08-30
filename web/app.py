from __future__ import annotations

import asyncio
import base64
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from motifgen.generation import MotifGenerator
from motifgen.tokenizer import MidiTokenizer, RecordedNote
from scripts.download_model import download_checkpoint


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "web" / "static"
DEFAULT_MODEL_PATH = ROOT / "artifacts" / "conditioned-best.pt"
MAX_UPLOAD_BYTES = 1_000_000
GENERATION_QUEUE_TIMEOUT_SECONDS = 120.0


def load_generator() -> tuple[MotifGenerator | None, str | None]:
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    model_url = os.getenv("MODEL_URL")
    model_sha256 = os.getenv("MODEL_SHA256")
    try:
        if not model_path.exists() and model_url:
            download_checkpoint(model_url, model_path, model_sha256)
        if not model_path.exists():
            return None, f"Checkpoint not found at {model_path}"
        return MotifGenerator.from_checkpoint(model_path), None
    except Exception as exc:
        return None, f"Checkpoint could not be loaded: {exc}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.generator, app.state.model_error = load_generator()
    app.state.generation_lock = asyncio.Lock()
    yield


app = FastAPI(
    title="Motif Piano Continuation API",
    description="Non-commercial motif-conditioned symbolic piano generation.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def index() -> HTMLResponse:
    public_base_url = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html.replace("__PUBLIC_BASE_URL__", public_base_url))


@app.get("/health")
async def health(request: Request) -> dict[str, object]:
    ready = request.app.state.generator is not None
    return {
        "status": "ok" if ready else "degraded",
        "model_ready": ready,
        "model_error": None if ready else request.app.state.model_error,
        "version": app.version,
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


async def motif_from_request(
    tokenizer: MidiTokenizer,
    motif_json: str | None,
    midi_file: UploadFile | None,
) -> list[int]:
    if bool(motif_json) == bool(midi_file):
        raise ValueError("Provide exactly one motif source: recorded notes or a MIDI file.")
    if motif_json:
        return tokenizer.notes_to_tokens(parse_recorded_notes(motif_json))

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
    return tokenizer.midi_bytes_to_tokens(payload)


@app.post("/api/generate")
async def generate(
    request: Request,
    motif_json: str | None = Form(default=None),
    midi_file: UploadFile | None = File(default=None),
    duration_seconds: int = Form(default=10),
    temperature: float = Form(default=1.0),
) -> dict[str, object]:
    generator: MotifGenerator | None = request.app.state.generator
    if generator is None:
        raise HTTPException(status_code=503, detail="The trained model is not configured on this service yet.")
    try:
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
            result = await asyncio.to_thread(
                generator.generate,
                motif_tokens,
                duration_seconds,
                temperature,
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
        }
    finally:
        lock.release()
