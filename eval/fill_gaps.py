"""
Download additional models to fill gaps and reach 500 total.
Downloads extra candidates to account for size filtering.
"""

import objaverse
import json
import os
import random
from pathlib import Path

MAX_SIZE_MB = 5.0
MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024
TARGET_TOTAL = 500
BUFFER_MULTIPLIER = 2.0  # Download 2x needed to account for size filtering


def main():
    # Load current state
    models_path = Path(__file__).parent / "models.json"
    gaps_path = Path(__file__).parent / "gaps.json"

    with open(models_path) as f:
        data = json.load(f)

    with open(gaps_path) as f:
        gaps = json.load(f)

    current_uids = {m["uid"] for m in data["models"]}
    print(f"Current: {len(data['models'])} models")
    print(f"Target:  {TARGET_TOTAL} models")

    # Collect UIDs to download (with buffer)
    to_download = []
    for cat, info in gaps["gaps"].items():
        needed = info["needed"]
        available_uids = [uid for uid in info["uids"] if uid not in current_uids]

        # Download extra to account for size filtering
        num_to_try = min(int(needed * BUFFER_MULTIPLIER), len(available_uids))
        selected = available_uids[:num_to_try]

        for uid in selected:
            to_download.append({"uid": uid, "category": cat, "needed": needed})

    print(f"\nDownloading {len(to_download)} candidate models...")

    # Download all at once
    uids_to_download = [m["uid"] for m in to_download]
    paths = objaverse.load_objects(uids_to_download, download_processes=4)

    # Filter by size and add to dataset
    added_per_cat = {cat: 0 for cat in gaps["gaps"].keys()}
    new_models = []
    deleted_count = 0

    for item in to_download:
        uid = item["uid"]
        cat = item["category"]
        needed = item["needed"]
        file_path = paths.get(uid, "")

        if not file_path or not os.path.exists(file_path):
            continue

        size_bytes = os.path.getsize(file_path)

        # Check if we still need models for this category and size is OK
        if added_per_cat[cat] < needed and size_bytes <= MAX_SIZE_BYTES:
            new_models.append({
                "uid": uid,
                "category": cat,
                "file_path": file_path
            })
            added_per_cat[cat] += 1
            size_mb = size_bytes / (1024 * 1024)
            print(f"  ✓ {cat}: {uid[:16]} ({size_mb:.1f} MB)")
        else:
            # Delete if too large or not needed
            os.remove(file_path)
            deleted_count += 1

    # Update models.json
    data["models"].extend(new_models)
    data["total"] = len(data["models"])

    # Recalculate total size
    total_size = 0
    for model in data["models"]:
        fp = model.get("file_path", "")
        if fp and os.path.exists(fp):
            total_size += os.path.getsize(fp)

    data["total_size_mb"] = round(total_size / (1024 * 1024), 1)

    with open(models_path, "w") as f:
        json.dump(data, f, indent=2)

    # Report
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Added:   {len(new_models)} models")
    print(f"Deleted: {deleted_count} (too large or not needed)")
    print(f"Total:   {data['total']} models ({data['total_size_mb']} MB)")

    # Show per-category results
    print(f"\nPer-category additions:")
    for cat, count in sorted(added_per_cat.items(), key=lambda x: -x[1]):
        if count > 0:
            needed = gaps["gaps"][cat]["needed"]
            status = "✓" if count >= needed else "partial"
            print(f"  {cat:40} +{count}/{needed} {status}")


if __name__ == "__main__":
    random.seed(42)
    main()
