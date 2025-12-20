"""
Objaverse model downloader.

Downloads 3D models from Objaverse by UID.
"""

import objaverse
from typing import Optional


def get_annotations(uids: list[str] = None) -> dict:
    """Load Objaverse annotations for specific UIDs (or all if None)."""
    return objaverse.load_annotations(uids)


def download_models(uids: list[str], download_processes: int = 4) -> dict:
    """
    Download models from Objaverse.

    Args:
        uids: List of Objaverse UIDs to download
        download_processes: Number of parallel downloads

    Returns:
        Dict mapping UID -> local file path
    """
    return objaverse.load_objects(uids, download_processes=download_processes)


def get_random_uids(count: int = 100, seed: Optional[int] = None) -> list[str]:
    """
    Get random UIDs from Objaverse.

    Args:
        count: Number of UIDs to return
        seed: Random seed for reproducibility

    Returns:
        List of UIDs
    """
    import random

    if seed is not None:
        random.seed(seed)

    uids = objaverse.load_uids()
    return random.sample(list(uids), min(count, len(uids)))
