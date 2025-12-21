"""Pre-download models during Docker build (no GPU required)."""

from huggingface_hub import snapshot_download


def download_florence():
    """Download Florence-2 weights (files only, no model loading)."""
    print("[download_models] Downloading Florence-2-base files...")
    snapshot_download(
        repo_id="microsoft/Florence-2-base",
        local_dir_use_symlinks=False
    )
    print("[download_models] Florence-2 files downloaded!")


def download_sentence_transformer():
    """Download sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    print("[download_models] Downloading all-mpnet-base-v2...")
    SentenceTransformer("all-mpnet-base-v2")
    print("[download_models] Sentence transformer downloaded!")


if __name__ == "__main__":
    download_florence()
    download_sentence_transformer()
