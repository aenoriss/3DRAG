"""
Download the 500 selected evaluation models.
"""

import objaverse
import json
import os
from pathlib import Path

def main():
    # Load selected UIDs
    models_path = Path(__file__).parent / "models.json"
    with open(models_path) as f:
        data = json.load(f)

    uids = [m["uid"] for m in data["models"]]
    print(f"Downloading {len(uids)} models...")

    # Download
    paths = objaverse.load_objects(uids, download_processes=4)

    # Calculate total size
    total_size = 0
    for uid, path in paths.items():
        if os.path.exists(path):
            total_size += os.path.getsize(path)

    print(f"\nDownloaded: {len(paths)} models")
    print(f"Total size: {total_size / (1024*1024):.1f} MB")

    # Update models.json with file paths
    uid_to_path = paths
    for model in data["models"]:
        model["file_path"] = uid_to_path.get(model["uid"], "")

    data["total_size_mb"] = round(total_size / (1024*1024), 1)

    with open(models_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Updated {models_path} with file paths")


if __name__ == "__main__":
    main()
