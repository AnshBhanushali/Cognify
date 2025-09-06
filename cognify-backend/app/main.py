import os
import uuid
import datetime
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import ImageProcessResponse, AudioProcessResponse
from app.models.image_embedder import save_label_to_chroma
from app.models.image_embedder import image_to_embedding, generate_detailed_label
from app.models.audio_transcriber import (
    transcribe_with_conf,
    detect_broken_words,
    summarize_transcript_simple,
)
from app.db import save_image_record, save_audio_record

# --------- paths ----------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
IMG_DIR = os.path.join(DATA_DIR, "uploads", "images")
AUD_DIR = os.path.join(DATA_DIR, "uploads", "audio")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(AUD_DIR, exist_ok=True)

# --------- app ----------
app = FastAPI(title="CognifyAI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


@app.get("/health")
def health():
    return {"status": "ok"}


# --------- Image: upload -> embedding + label ----------
@app.post("/upload/image", response_model=ImageProcessResponse)
async def upload_image(file: UploadFile = File(...), include_embedding: bool = True):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    # Save
    uid = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] or ".png"
    safe_name = f"{uid}{ext}"
    dest_path = os.path.join(IMG_DIR, safe_name)
    with open(dest_path, "wb") as f:
        f.write(await file.read())

    # Embed + Predict label
    embedding = image_to_embedding(dest_path)
    predicted_label, suggestions = generate_detailed_label(dest_path, embedding)
    ts = _now_iso()

    # Persist record
    save_image_record(uid, file.filename, ts, embedding)

    return ImageProcessResponse(
        id=uid,
        filename=file.filename,
        timestamp=ts,
        embedding_dim=len(embedding),
        embedding=embedding if include_embedding else None,
        predicted_label=predicted_label,
        suggestions=suggestions,
    )

@app.post("/confirm")
async def confirm(image_id: str, label: str, embedding: List[float]):
    """
    Save a user-confirmed label into ChromaDB so the system can remember it.
    """
    save_label_to_chroma(image_id, label, embedding)
    return {"ok": True, "saved_label": label}

# --------- Audio: upload -> transcript + broken detection ----------
@app.post("/upload/audio", response_model=AudioProcessResponse)
async def upload_audio(file: UploadFile = File(...), summarize_if_clean: bool = True):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    if not (file.content_type or "").startswith("audio/"):
        raise HTTPException(status_code=400, detail="Please upload an audio file")

    uid = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] or ".wav"
    safe_name = f"{uid}{ext}"
    dest_path = os.path.join(AUD_DIR, safe_name)
    with open(dest_path, "wb") as f:
        f.write(await file.read())

    transcript, avg_conf, word_probs, duration = transcribe_with_conf(dest_path)
    broken_words = detect_broken_words(word_probs, prob_threshold=0.55)

    # Heuristic: mark audio as "broken" if enough uncertain tokens OR very low avg confidence
    is_broken = (avg_conf < 0.45) or (
        len(word_probs) > 0 and (len(broken_words) / max(1, len(word_probs)) > 0.20)
    )

    summary = None
    if summarize_if_clean and not is_broken:
        summary = summarize_transcript_simple(transcript, max_words=24)

    ts = _now_iso()
    save_audio_record(
        uid, file.filename, ts, duration, transcript, avg_conf, is_broken, broken_words
    )

    

    return AudioProcessResponse(
        id=uid,
        filename=file.filename,
        timestamp=ts,
        duration_sec=duration,
        transcript=transcript,
        avg_confidence=avg_conf,
        is_broken=is_broken,
        broken_words=broken_words,
        summary=summary,
    )
