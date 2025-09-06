import torch
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import re

# Load CLIP model for embeddings
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load BLIP model for captions (fine-grained labels)
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

def image_to_embedding(image_path: str):
    """Generate a CLIP embedding for the image."""
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
    return embedding.squeeze().tolist()

def generate_detailed_label(image_path: str) -> str:
    """Generate a descriptive label for the image using BLIP captioning with cleanup."""
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(image, return_tensors="pt")
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_length=30)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)

    # Step 1: lowercase + clean punctuation
    caption = caption.lower().strip()
    caption = re.sub(r"[^a-z0-9\s]", "", caption)

    # Step 2: remove repeated consecutive words ("cater cater cater" → "cater")
    words = caption.split()
    cleaned_words = []
    for w in words:
        if not cleaned_words or cleaned_words[-1] != w:
            cleaned_words.append(w)

    # Step 3: take only first 4–6 words for compact label
    cleaned_words = cleaned_words[:6]

    # Step 4: title case and dash-join
    structured_label = "-".join([w.capitalize() for w in cleaned_words])

    return structured_label
