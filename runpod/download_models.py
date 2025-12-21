"""Pre-download models during Docker build (no GPU required)."""

from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports


def _fixed_get_imports(filename: str) -> list:
    """Patch to remove flash_attn from imports."""
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


def download_florence():
    """Download Florence-2 weights (CPU only, no GPU needed)."""
    from transformers import AutoProcessor, AutoModelForCausalLM

    print("Downloading Florence-2-base...")
    with patch("transformers.dynamic_module_utils.get_imports", _fixed_get_imports):
        AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
        AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
    print("Florence-2 downloaded!")


def download_sentence_transformer():
    """Download sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    print("Downloading all-mpnet-base-v2...")
    SentenceTransformer("all-mpnet-base-v2")
    print("Sentence transformer downloaded!")


if __name__ == "__main__":
    download_florence()
    download_sentence_transformer()
