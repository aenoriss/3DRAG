"""
Analyze which categories need more models to reach 500 total.
Check LVIS for available small models to fill gaps.
"""

import objaverse
import json
import os
from pathlib import Path

TARGET_PER_CATEGORY = 10
TARGET_TOTAL = 500
MAX_SIZE_MB = 5.0


def main():
    # Load current models
    models_path = Path(__file__).parent / "models.json"
    with open(models_path) as f:
        data = json.load(f)

    # Count per category
    current_counts = {}
    current_uids = set()
    for model in data["models"]:
        cat = model["category"]
        current_counts[cat] = current_counts.get(cat, 0) + 1
        current_uids.add(model["uid"])

    # Calculate gaps
    gaps = {}
    for cat in data["categories"]:
        current = current_counts.get(cat, 0)
        needed = TARGET_PER_CATEGORY - current
        if needed > 0:
            gaps[cat] = {"current": current, "needed": needed}

    total_needed = TARGET_TOTAL - len(data["models"])

    print(f"Current: {len(data['models'])} models")
    print(f"Target:  {TARGET_TOTAL} models")
    print(f"Need:    {total_needed} more models")
    print(f"\n{'='*60}")
    print(f"CATEGORIES NEEDING MORE MODELS")
    print(f"{'='*60}")

    # Sort by most needed first
    sorted_gaps = sorted(gaps.items(), key=lambda x: -x[1]["needed"])

    for cat, info in sorted_gaps:
        bar = "█" * info["current"] + "░" * info["needed"]
        print(f"{cat:40} {info['current']:2}/10 {bar} (need {info['needed']})")

    # Load LVIS to check available models
    print(f"\n{'='*60}")
    print(f"CHECKING LVIS FOR AVAILABLE SMALL MODELS...")
    print(f"{'='*60}")

    lvis = objaverse.load_lvis_annotations()

    # For each gap category, check how many models are available (excluding already selected)
    can_fill = {}
    for cat, info in sorted_gaps:
        available_uids = [uid for uid in lvis.get(cat, []) if uid not in current_uids]
        can_fill[cat] = {
            "needed": info["needed"],
            "available": len(available_uids),
            "uids": available_uids[:20]  # Keep first 20 for potential download
        }
        status = "✓" if len(available_uids) >= info["needed"] else "⚠"
        print(f"{status} {cat:40} need {info['needed']:2}, available: {len(available_uids)}")

    # Summary
    total_can_add = sum(min(v["needed"], v["available"]) for v in can_fill.values())
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Can potentially add: {total_can_add} models (before size filtering)")
    print(f"After ~30% size filter: ~{int(total_can_add * 0.7)} models")

    # Save gap analysis for next script
    output = {
        "current_total": len(data["models"]),
        "target_total": TARGET_TOTAL,
        "gaps": can_fill
    }

    gap_path = Path(__file__).parent / "gaps.json"
    with open(gap_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved gap analysis to {gap_path}")


if __name__ == "__main__":
    main()
