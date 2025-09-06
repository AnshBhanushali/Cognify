from typing import Optional, List, Tuple
import torch, torch.nn.functional as F
from transformers import (
    CLIPProcessor,
    CLIPModel,
    BlipProcessor,
    BlipForConditionalGeneration,
)
from PIL import Image
import chromadb

# ----------------- Init Models -----------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP for embeddings + zero-shot classification
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# BLIP for captions (optional fallback)
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# ----------------- Init Chroma -----------------
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("image_labels")

# ----------------- Label Set (for zero-shot) -----------------
GENERIC_LABELS = [
    "car",
    "dog",
    "cat",
    "person",
    "bicycle",
    "tree",
    "building",
    "computer",
    "phone",
    "shoe",
]
TEXT_PROMPTS = [f"a photo of a {label}" for label in GENERIC_LABELS]


# ----------------- Embedding -----------------
def image_to_embedding(image_path: str) -> List[float]:
    """Generate a CLIP embedding for the image."""
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
        embedding = F.normalize(embedding, dim=-1)
    return embedding.squeeze().tolist()


# ----------------- Zero-Shot -----------------
def zero_shot_labels(image_path: str, topk: int = 3) -> List[dict]:
    """Classify against a generic label set using CLIP zero-shot."""
    image = Image.open(image_path).convert("RGB")
    img_inputs = clip_processor(images=image, return_tensors="pt").to(device)
    txt_inputs = clip_processor(text=TEXT_PROMPTS, return_tensors="pt", padding=True).to(
        device
    )

    with torch.no_grad():
        img_feats = F.normalize(clip_model.get_image_features(**img_inputs), dim=-1)
        txt_feats = F.normalize(clip_model.get_text_features(**txt_inputs), dim=-1)
        sims = (img_feats @ txt_feats.T).squeeze(0)
        vals, idx = torch.topk(sims, k=min(topk, len(TEXT_PROMPTS)))

    return [{"label": GENERIC_LABELS[i], "score": float(vals[j])} for j, i in enumerate(idx)]


# ----------------- BLIP -----------------
def blip_caption(image_path: str) -> str:
    """Optional: generate a descriptive caption using BLIP."""
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_length=30)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption


# ----------------- Label Generation -----------------
def generate_detailed_label(
    path: str, embedding: Optional[List[float]] = None
) -> Tuple[str, List[dict]]:
    """Predict a label using Chroma (retrieval), CLIP (zero-shot), or BLIP (fallback)."""
    if embedding is None:
        embedding = image_to_embedding(path)

    # 1) Retrieval-first from Chroma
    results = collection.query(query_embeddings=[embedding], n_results=1)
    if results and results.get("documents") and len(results["documents"][0]) > 0:
        dist_list = results.get("distances")
        if dist_list and len(dist_list[0]) > 0:
            dist = dist_list[0][0]
            if dist <= 0.70:  # threshold
                return results["documents"][0][0], []

    # 2) Zero-shot fallback
    suggestions = zero_shot_labels(path)
    if suggestions and suggestions[0]["score"] >= 0.22:
        return suggestions[0]["label"], suggestions

    # 3) BLIP fallback
    caption = blip_caption(path)
    return caption, []


# ----------------- Save to Chroma -----------------
def save_label_to_chroma(image_id: str, label: str, embedding: List[float]):
    """Persist user-confirmed labels into ChromaDB."""
    collection.add(
        ids=[image_id],
        embeddings=[embedding],
        documents=[label],
        metadatas=[{"source": "user"}],
    )
