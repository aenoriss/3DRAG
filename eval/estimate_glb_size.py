"""
Estimate average GLB file size from Objaverse-LVIS models.
Downloads a sample and measures actual file sizes.
"""

import objaverse
import os
import random
from pathlib import Path

SAMPLE_SIZE = 50  # Download 50 models to estimate


def main():
    print("Loading LVIS annotations...")
    lvis = objaverse.load_lvis_annotations()

    # Get diverse sample from different categories
    all_uids = []
    categories = list(lvis.keys())
    random.shuffle(categories)

    for cat in categories[:20]:  # 20 categories
        uids = lvis[cat]
        if uids:
            all_uids.extend(random.sample(uids, min(5, len(uids))))

    # Take final sample
    sample_uids = random.sample(all_uids, min(SAMPLE_SIZE, len(all_uids)))
    print(f"\nDownloading {len(sample_uids)} sample models...")

    # Download
    paths = objaverse.load_objects(sample_uids, download_processes=4)

    # Measure sizes
    sizes = []
    for uid, path in paths.items():
        if os.path.exists(path):
            size_bytes = os.path.getsize(path)
            size_mb = size_bytes / (1024 * 1024)
            sizes.append(size_mb)
            print(f"  {uid[:16]}: {size_mb:.2f} MB")

    if not sizes:
        print("No files downloaded!")
        return

    # Statistics
    avg = sum(sizes) / len(sizes)
    min_size = min(sizes)
    max_size = max(sizes)
    median = sorted(sizes)[len(sizes) // 2]

    print(f"\n{'='*40}")
    print(f"RESULTS ({len(sizes)} models)")
    print(f"{'='*40}")
    print(f"Average:  {avg:.2f} MB")
    print(f"Median:   {median:.2f} MB")
    print(f"Min:      {min_size:.2f} MB")
    print(f"Max:      {max_size:.2f} MB")
    print(f"Total:    {sum(sizes):.2f} MB")

    print(f"\n{'='*40}")
    print("ESTIMATES")
    print(f"{'='*40}")
    for count in [100, 500, 1000, 5000]:
        total = count * avg
        print(f"{count:5} models: {total:8.1f} MB ({total/1024:.2f} GB)")


if __name__ == "__main__":
    random.seed(42)
    main()
