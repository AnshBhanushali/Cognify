import os
import uuid
import datetime
from typing import List
from pydantic import BaseModel

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app.schemas import ImageProcessResponse, AudioProcessResponse
from app.models.image_embedder import (
    image_to_embedding,
    generate_detailed_label,
    save_label_to_chroma,
    collection,
)
from app.models.audio_transcriber import (
    transcribe_with_conf,
    detect_broken_words,
    summarize_transcript_simple,
)
from app.db import save_image_record, save_audio_record
import io
import csv

class ConfirmRequest(BaseModel):
    image_id: str
    label: str
    embedding: List[float]

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

    # Persist record (optional to your db layer)
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


# --------- Image: confirm -> save to Chroma ----------
@app.post("/confirm")
async def confirm(req: ConfirmRequest):
    save_label_to_chroma(req.image_id, req.label, req.embedding)
    return {"ok": True, "saved_label": req.label}


# --------- Dataset: list saved labels ----------
@app.get("/dataset")
async def list_dataset(limit: int = 50):
    results = collection.get()
    data = []
    for i, doc in enumerate(results["documents"]):
        data.append({
            "id": results["ids"][i],
            "label": doc,
            "metadata": results["metadatas"][i],
        })
    return {"count": len(data), "items": data[:limit]}


# --------- Dataset: download as JSON ----------
@app.get("/dataset/download/json")
async def download_dataset_json():
    results = collection.get()
    data = []
    for i, doc in enumerate(results["documents"]):
        data.append({
            "id": results["ids"][i],
            "label": doc,
            "metadata": results["metadatas"][i],
        })
    return JSONResponse(content=data)


# --------- Dataset: download as CSV ----------
@app.get("/dataset/download/csv")
async def download_dataset_csv():
    results = collection.get()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "label", "metadata"])

    for i, doc in enumerate(results["documents"]):
        writer.writerow([results["ids"][i], doc, results["metadatas"][i]])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=dataset.csv"},
    )


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
