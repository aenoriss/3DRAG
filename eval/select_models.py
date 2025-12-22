"""
Step 1: Select 500 models from 50 LVIS categories (10 each).
Only saves UIDs - downloads happen during processing.
"""

import objaverse
import json
import random
from pathlib import Path

# 50 diverse LVIS categories × 10 models = 500 models
CATEGORIES = [
    # Vehicles (5)
    "race_car", "helicopter", "airplane", "bicycle", "pickup_truck",
    # Animals (10)
    "lion", "rabbit", "elephant", "owl", "teddy_bear",
    "shark", "wolf", "frog", "penguin", "butterfly",
    # Furniture (5)
    "chair", "armchair", "table", "bookcase", "chandelier",
    # Food (5)
    "banana", "apple", "doughnut", "mushroom", "pumpkin",
    # Electronics (4)
    "telephone", "television_set", "computer_keyboard",
    "monitor_(computer_equipment) computer_monitor",
    # Weapons/Combat (5)
    "sword", "gun", "rifle", "shield", "armor",
    # Sports/Music (4)
    "soccer_ball", "skateboard", "guitar", "piano",
    # Clothing/Accessories (4)
    "shoe", "sunglasses", "necklace", "ring",
    # Objects/Decorative (5)
    "vase", "sculpture", "figurine", "mask", "globe",
    # Outdoor/Nature (3)
    "windmill", "fireplug", "lamppost",
]

MODELS_PER_CATEGORY = 10


def main():
    print(f"Target: {len(CATEGORIES)} categories × {MODELS_PER_CATEGORY} = {len(CATEGORIES) * MODELS_PER_CATEGORY} models")

    print("\nLoading LVIS annotations...")
    lvis = objaverse.load_lvis_annotations()

    # Verify categories
    missing = [c for c in CATEGORIES if c not in lvis]
    if missing:
        print(f"ERROR: Missing categories: {missing}")
        return

    # Select UIDs (no download yet)
    models = []
    for cat in CATEGORIES:
        uids = lvis[cat]
        sampled = random.sample(uids, min(MODELS_PER_CATEGORY, len(uids)))
        for uid in sampled:
            models.append({"uid": uid, "category": cat})
        print(f"  {cat}: {len(sampled)}/{len(uids)}")

    # Save
    output = {
        "total": len(models),
        "num_categories": len(CATEGORIES),
        "models_per_category": MODELS_PER_CATEGORY,
        "categories": CATEGORIES,
        "models": models
    }

    output_path = Path(__file__).parent / "models.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*50}")
    print(f"SAVED: {output_path}")
    print(f"Total: {len(models)} models from {len(CATEGORIES)} categories")
    print("(GLB files will be downloaded during processing)")


if __name__ == "__main__":
    random.seed(42)
    main()
