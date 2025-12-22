"""
Download models targeting 500 total (10 per category).
"""

import objaverse
import json
import os
import sys
from pathlib import Path

MAX_SIZE_BYTES = 5 * 1024 * 1024  # 5MB
TARGET_PER_CAT = 10


def main():
    models_path = Path(__file__).parent / "models.json"

    with open(models_path) as f:
        data = json.load(f)

    print("Loading LVIS...", flush=True)
    lvis = objaverse.load_lvis_annotations()
    print("LVIS loaded.\n", flush=True)

    categories = data["categories"]
    all_models = []

    for i, cat in enumerate(categories):
        print(f"[{i+1}/{len(categories)}] {cat}: ", end="", flush=True)

        available_uids = lvis.get(cat, [])
        good_models = []
        tried = 0

        for uid in available_uids:
            if len(good_models) >= TARGET_PER_CAT:
                break

            tried += 1
            print(".", end="", flush=True)  # Show progress

            try:
                paths = objaverse.load_objects([uid], download_processes=1)
                file_path = paths.get(uid, "")

                if not file_path or not os.path.exists(file_path):
                    continue

                size = os.path.getsize(file_path)

                if size <= MAX_SIZE_BYTES:
                    good_models.append({
                        "uid": uid,
                        "category": cat,
                        "file_path": file_path
                    })
                    print(f"✓", end="", flush=True)
                else:
                    os.remove(file_path)
                    print(f"✗", end="", flush=True)

            except Exception as e:
                print(f"!", end="", flush=True)
                continue

        all_models.extend(good_models)
        print(f" {len(good_models)}/10 (tried {tried})", flush=True)

    # Save
    total_size = sum(os.path.getsize(m["file_path"]) for m in all_models if os.path.exists(m.get("file_path", "")))
    data["models"] = all_models
    data["total"] = len(all_models)
    data["total_size_mb"] = round(total_size / (1024 * 1024), 1)

    with open(models_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n{'='*40}")
    print(f"Total: {len(all_models)} models ({data['total_size_mb']} MB)")


if __name__ == "__main__":
    main()
