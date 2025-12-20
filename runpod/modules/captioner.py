"""
Florence-2 image captioning module.

Generates descriptions of 3D model renders for semantic search.
"""

import re
import torch
from PIL import Image
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

# Global model references
FLORENCE_MODEL = None
FLORENCE_PROCESSOR = None


def _fixed_get_imports(filename: str) -> list:
    """Patch to remove flash_attn from imports (not actually needed)."""
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


def load_florence(model_id: str = "microsoft/Florence-2-base"):
    """
    Load Florence-2 model.

    Args:
        model_id: HuggingFace model ID (base or large)
    """
    global FLORENCE_MODEL, FLORENCE_PROCESSOR

    if FLORENCE_MODEL is not None:
        return  # Already loaded

    from transformers import AutoProcessor, AutoModelForCausalLM

    print(f"Loading Florence-2 from {model_id}...")

    with patch("transformers.dynamic_module_utils.get_imports", _fixed_get_imports):
        FLORENCE_PROCESSOR = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        FLORENCE_MODEL = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="eager"
        ).to("cuda")

    print("Florence-2 loaded!")


def clean_caption(caption: str) -> str:
    """
    Clean up caption for better embedding quality.

    Removes filler phrases to get direct, searchable descriptions.
    """
    # Remove task tokens
    for token in ["<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<CAPTION>"]:
        caption = caption.replace(token, "")

    # Remove common filler phrases
    filler_patterns = [
        r"^the image (shows?|contains?|depicts?|features?|displays?)\s*",
        r"^this image (shows?|contains?|depicts?|features?|displays?)\s*",
        r"^in this image,?\s*",
        r"^i (can )?see\s*",
        r"^there (is|are)\s*",
        r"^we (can )?see\s*",
        r"^the picture (shows?|contains?)\s*",
    ]

    for pattern in filler_patterns:
        caption = re.sub(pattern, "", caption, flags=re.IGNORECASE)

    caption = " ".join(caption.split())
    if caption:
        caption = caption[0].upper() + caption[1:]

    return caption


def caption_image(image: Image.Image) -> str:
    """
    Generate caption for a single image.

    Args:
        image: PIL Image

    Returns:
        Cleaned caption string
    """
    global FLORENCE_MODEL, FLORENCE_PROCESSOR

    if FLORENCE_MODEL is None:
        load_florence()

    task = "<MORE_DETAILED_CAPTION>"

    inputs = FLORENCE_PROCESSOR(
        text=task,
        images=image,
        return_tensors="pt"
    ).to("cuda", torch.float16)

    with torch.no_grad():
        outputs = FLORENCE_MODEL.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=1,
            do_sample=False
        )

    caption = FLORENCE_PROCESSOR.batch_decode(outputs, skip_special_tokens=True)[0]
    return clean_caption(caption)


def caption_images_batch(images: list[Image.Image]) -> list[str]:
    """
    Batch caption multiple images.

    Args:
        images: List of PIL Images

    Returns:
        List of cleaned captions
    """
    global FLORENCE_MODEL, FLORENCE_PROCESSOR

    if not images:
        return []

    if FLORENCE_MODEL is None:
        load_florence()

    task = "<MORE_DETAILED_CAPTION>"

    inputs = FLORENCE_PROCESSOR(
        text=[task] * len(images),
        images=images,
        return_tensors="pt",
        padding=True
    ).to("cuda", torch.float16)

    with torch.no_grad():
        outputs = FLORENCE_MODEL.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=1,
            do_sample=False
        )

    captions = FLORENCE_PROCESSOR.batch_decode(outputs, skip_special_tokens=True)
    return [clean_caption(c) for c in captions]
