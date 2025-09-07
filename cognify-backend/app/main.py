import os
import uuid
import datetime
import time
import io
import csv
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from dotenv import load_dotenv
from openai import OpenAI

# --------- load environment ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")  # optional for Hugging Face
client = OpenAI(api_key=OPENAI_API_KEY)

# --------- local imports ----------
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
    denoise_audio,
    audio_embedding_mfcc,
    diarize_energy_fallback,
)
from app.models.audio_memory import save_audio_label_to_chroma, query_audio_label
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


# --------- Global ML state (Sprint 2) ----------
X_train: List[np.ndarray] = []
y_train: List[str] = []
knn_model: Optional[KNeighborsClassifier] = None


def retrain_knn():
    global knn_model
    if not X_train:
        return None
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)
    return knn_model


# --------- LLM Hierarchy Suggestion ----------
def suggest_hierarchy(label: str) -> dict:
    """
    Uses OpenAI GPT to expand a flat label into a hierarchy.
    Example: "Civic" -> {"root": "Car", "make": "Honda", "model": "Civic"}
    """
    if not OPENAI_API_KEY:
        return {"error": "Missing OPENAI_API_KEY in .env"}

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI that classifies labels into hierarchies."},
                {"role": "user", "content": f"Organize the label '{label}' into a hierarchy. "
                                             "Return JSON with fields like root, category, subcategory, etc. "
                                             "Be generic so it works for animals, vehicles, objects, etc."}
            ],
            response_format={"type": "json_object"},
            max_tokens=200,
        )
        return resp.choices[0].message.parsed
    except Exception as e:
        return {"error": str(e)}


# --------- Image: upload -> embedding + label ----------
@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...), include_embedding: bool = True):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    # save file
    uid = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] or ".png"
    safe_name = f"{uid}{ext}"
    dest_path = os.path.join(IMG_DIR, safe_name)
    with open(dest_path, "wb") as f:
        f.write(await file.read())

    ts = _now_iso()

    # --- Step 1: Embedding ---
    t0 = time.perf_counter()
    embedding = image_to_embedding(dest_path)
    embedding_time = round(time.perf_counter() - t0, 3)

    # --- Step 2: ChromaDB search ---
    t1 = time.perf_counter()
    predicted_label, suggestions = generate_detailed_label(dest_path, embedding)
    chroma_time = round(time.perf_counter() - t1, 3)

    top_conf = suggestions[0]["score"] if suggestions else 0.0

    # --- Step 3: OpenAI fallback if confidence is weak ---
    llm_label = None
    hierarchy = None
    if top_conf < 0.7:  # threshold can be tuned
        try:
            llm_resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert classifier. Expand generic labels into more specific hierarchies."},
                    {"role": "user", "content": f"The CLIP embedding suggests '{predicted_label}'. "
                                                 "Refine it to be more specific. Example: 'Car → Honda → Civic'. "
                                                 "If it's an object or animal, give the most detailed classification you can. "
                                                 "Return JSON with fields like root, category, subcategory, etc."}
                ],
                response_format={"type": "json_object"},
                max_tokens=200,
            )
            parsed = llm_resp.choices[0].message.parsed
            if isinstance(parsed, dict) and "root" in parsed:
                hierarchy = parsed
                llm_label = " → ".join(parsed.values())
            else:
                llm_label = llm_resp.choices[0].message.content.strip()
        except Exception as e:
            print("OpenAI failed:", e)

        if llm_label:
            suggestions.append({
                "label": llm_label,
                "score": 0.8,
                "source": "openai",
                "hierarchy": hierarchy
            })

    # --- Step 4: Save record ---
    save_image_record(uid, file.filename, ts, embedding)

    # --- Step 5: Return response ---
    return {
        "id": uid,
        "filename": file.filename,
        "timestamp": ts,
        "embedding_dim": len(embedding),
        "embedding": embedding if include_embedding else None,
        "predicted_label": predicted_label,
        "suggestions": suggestions,
        "stats": {
            "embedding_time": embedding_time,
            "chroma_time": chroma_time,
            "similar_labels": len(suggestions),
            "top_confidence": round(top_conf, 3),
        },
    }


# --------- Image: confirm ----------
@app.post("/confirm")
async def confirm(image_id: str, label: str, embedding: List[float]):
    save_label_to_chroma(image_id, label, embedding)

    # also update KNN dataset
    X_train.append(np.array(embedding))
    y_train.append(label)
    retrain_knn()

    # also call LLM for hierarchy
    hierarchy = suggest_hierarchy(label)

    return {"ok": True, "saved_label": label, "hierarchy": hierarchy}


# --------- Dataset: list ----------
@app.get("/dataset")
async def list_dataset(limit: int = 50):
    results = collection.get()
    data = []
    for i, doc in enumerate(results["documents"]):
        data.append(
            {
                "id": results["ids"][i],
                "label": doc,
                "metadata": results["metadatas"][i],
            }
        )
    return {"count": len(data), "items": data[:limit]}


# --------- Dataset: download JSON ----------
@app.get("/dataset/download/json")
async def download_dataset_json():
    results = collection.get()
    data = []
    for i, doc in enumerate(results["documents"]):
        data.append(
            {
                "id": results["ids"][i],
                "label": doc,
                "metadata": results["metadatas"][i],
            }
        )
    return JSONResponse(content=data)


# --------- Dataset: download CSV ----------
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


# --------- Audio: upload -> transcript + extras ----------
@app.post("/upload/audio")
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

    transcript, avg_conf, words, duration, segs = transcribe_with_conf(dest_path)
    broken_words, broken_spans = detect_broken_words(words, prob_threshold=0.55)

    is_broken = (avg_conf < 0.45) or (
        len(words) > 0 and (len(broken_words) / max(1, len(words)) > 0.20)
    )

    summary = None
    if summarize_if_clean and not is_broken:
        summary = summarize_transcript_simple(transcript, max_words=24)

    diarization = diarize_energy_fallback(dest_path)

    ts = _now_iso()
    save_audio_record(uid, file.filename, ts, duration, transcript, avg_conf, is_broken, broken_words)

    embedding = audio_embedding_mfcc(dest_path)

    retrieval = query_audio_label(embedding, n_results=1)
    retrieved_label = None
    if retrieval and retrieval.get("documents") and retrieval["documents"][0]:
        retrieved_label = retrieval["documents"][0][0]

    return {
        "id": uid,
        "filename": file.filename,
        "timestamp": ts,
        "duration_sec": duration,
        "transcript": transcript,
        "avg_confidence": avg_conf,
        "is_broken": is_broken,
        "broken_words": broken_words,
        "summary": summary,
        "words": words,
        "segments": segs,
        "broken_spans": broken_spans,
        "diarization": diarization,
        "audio_embedding": embedding,
        "retrieved_label": retrieved_label,
    }


# --------- Audio: confirm ----------
class ConfirmAudioReq(BaseModel):
    audio_id: str
    label: str
    embedding: List[float]


@app.post("/confirm_audio")
async def confirm_audio(req: ConfirmAudioReq):
    save_audio_label_to_chroma(req.audio_id, req.label, req.embedding)

    # also update classifier
    X_train.append(np.array(req.embedding))
    y_train.append(req.label)
    retrain_knn()

    # also call LLM for hierarchy
    hierarchy = suggest_hierarchy(req.label)

    return {"ok": True, "saved_label": req.label, "hierarchy": hierarchy}


# --------- Audio: repair (denoise) ----------
@app.get("/audio/{audio_id}/repair")
async def repair_audio(audio_id: str):
    matches = [fn for fn in os.listdir(AUD_DIR) if fn.startswith(audio_id)]
    if not matches:
        raise HTTPException(status_code=404, detail="audio not found")
    src = os.path.join(AUD_DIR, matches[0])

    out_name = f"{audio_id}_denoised.wav"
    out_path = os.path.join(AUD_DIR, out_name)
    meta = denoise_audio(src, out_path)

    headers = {
        "X-SNR-Before": str(round(meta.get("snr_before", 0.0), 2)),
        "X-SNR-After": str(round(meta.get("snr_after", 0.0), 2)),
    }
    return FileResponse(out_path, media_type="audio/wav", filename=out_name, headers=headers)


# --------- Sprint 2: Classifier Endpoints ----------
class PredictReq(BaseModel):
    embedding: List[float]

@app.post("/predict")
async def predict(req: PredictReq):
    if not knn_model:
        return {"label": None, "confidence": 0}
    emb = np.array(req.embedding).reshape(1, -1)
    pred = knn_model.predict(emb)[0]
    conf = float(np.max(knn_model.predict_proba(emb)))
    return {"label": pred, "confidence": conf}


@app.post("/retrain")
async def retrain():
    if not X_train:
        return {"status": "no data"}
    retrain_knn()
    return {"status": "retrained", "samples": len(X_train)}


@app.get("/embeddings/all")
async def get_embeddings():
    return {"X": [x.tolist() for x in X_train], "y": y_train}


# --------- Standalone hierarchy endpoint ----------
@app.post("/hierarchy")
async def hierarchy(label: str):
    return suggest_hierarchy(label)
