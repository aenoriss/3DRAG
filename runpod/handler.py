import runpod
from transformers import AutoModel, AutoProcessor
from PIL import Image
import torch
import base64
import io
import numpy as np

# Load model at cold start (stays in memory for warm requests)
print("Loading SigLIP2 model...")
model = AutoModel.from_pretrained("google/siglip2-so400m-patch14-384").cuda()
processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch14-384")
model.eval()
print("Model loaded!")


def embed_text(texts: list[str]) -> list[list[float]]:
    """Embed one or more text strings."""
    inputs = processor(text=texts, return_tensors="pt", padding=True).cuda()
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().tolist()


def embed_images(images_b64: list[str]) -> list[list[float]]:
    """Embed one or more base64-encoded images."""
    images = []
    for img_b64 in images_b64:
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        images.append(img)

    inputs = processor(images=images, return_tensors="pt").cuda()
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().tolist()


def handler(event):
    """
    RunPod serverless handler.

    Input formats:
    - {"text": "a red chair"}                    → single text embedding
    - {"texts": ["chair", "table"]}              → batch text embeddings
    - {"image": "<base64>"}                      → single image embedding
    - {"images": ["<base64>", "<base64>"]}       → batch image embeddings
    """
    try:
        input_data = event.get("input", {})

        # Single text
        if "text" in input_data:
            embeddings = embed_text([input_data["text"]])
            return {"embedding": embeddings[0]}

        # Batch texts
        if "texts" in input_data:
            embeddings = embed_text(input_data["texts"])
            return {"embeddings": embeddings}

        # Single image
        if "image" in input_data:
            embeddings = embed_images([input_data["image"]])
            return {"embedding": embeddings[0]}

        # Batch images
        if "images" in input_data:
            embeddings = embed_images(input_data["images"])
            return {"embeddings": embeddings}

        return {"error": "No valid input. Use 'text', 'texts', 'image', or 'images'."}

    except Exception as e:
        return {"error": str(e)}


# Start the serverless worker
runpod.serverless.start({"handler": handler})
