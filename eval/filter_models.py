"""
Filter models to keep only those ≤5MB to stay under 1GB total.
Deletes large GLB files and updates models.json.
"""

import json
import os
from pathlib import Path

MAX_SIZE_MB = 5.0
MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024


def main():
    models_path = Path(__file__).parent / "models.json"

    with open(models_path) as f:
        data = json.load(f)

    print(f"Original: {len(data['models'])} models")

    # Filter and track stats
    kept_models = []
    removed_models = []
    kept_size = 0
    removed_size = 0
    category_counts = {}

    for model in data["models"]:
        file_path = model.get("file_path", "")

        if not file_path or not os.path.exists(file_path):
            # No file, skip
            removed_models.append(model)
            continue

        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)

        if size_bytes <= MAX_SIZE_BYTES:
            kept_models.append(model)
            kept_size += size_bytes
            cat = model["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1
        else:
            removed_models.append(model)
            removed_size += size_bytes
            # Delete the large file
            os.remove(file_path)
            print(f"  Deleted: {Path(file_path).name} ({size_mb:.1f} MB)")

    # Update models.json
    data["models"] = kept_models
    data["total"] = len(kept_models)
    data["total_size_mb"] = round(kept_size / (1024 * 1024), 1)

    with open(models_path, "w") as f:
        json.dump(data, f, indent=2)

    # Report
    print(f"\n{'='*50}")
    print(f"FILTERING COMPLETE")
    print(f"{'='*50}")
    print(f"Kept:    {len(kept_models)} models ({kept_size / (1024*1024):.1f} MB)")
    print(f"Removed: {len(removed_models)} models ({removed_size / (1024*1024):.1f} MB)")

    # Category breakdown
    print(f"\nModels per category:")
    for cat in sorted(category_counts.keys()):
        count = category_counts[cat]
        bar = "█" * count
        print(f"  {cat:40} {count:2} {bar}")

    # Categories with few models
    low_cats = [c for c, n in category_counts.items() if n < 5]
    if low_cats:
        print(f"\nCategories with <5 models: {len(low_cats)}")
        for c in low_cats:
            print(f"  - {c}: {category_counts[c]}")


if __name__ == "__main__":
    main()
