"""Pre-download Florence-2 model with flash_attn import patch."""
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports


def fixed_get_imports(filename: str) -> list:
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    from transformers import AutoProcessor, AutoModelForCausalLM

    model_id = "microsoft/Florence-2-base"
    print(f"Downloading {model_id}...")

    AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

    print("Done!")
