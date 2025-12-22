"""
Generate initial ground truth by mapping queries to models in their category.
This creates a baseline where all models in a category are considered relevant.

For more accurate evaluation, manually refine ground_truth.json:
- Remove models that don't match the specific query
- Add relevance scores (1-3) for graded evaluation
"""

import json
from pathlib import Path


def main():
    eval_dir = Path(__file__).parent

    # Load queries
    with open(eval_dir / "queries.json") as f:
        queries_data = json.load(f)

    # Load models
    with open(eval_dir / "models.json") as f:
        models_data = json.load(f)

    # Group models by category
    models_by_category = {}
    for model in models_data["models"]:
        cat = model["category"]
        if cat not in models_by_category:
            models_by_category[cat] = []
        models_by_category[cat].append(model["uid"])

    # Generate ground truth
    ground_truth = {}

    for query in queries_data["queries"]:
        qid = query["id"]
        category = query["category"]

        # All models in the category are considered relevant
        relevant_uids = models_by_category.get(category, [])
        ground_truth[qid] = relevant_uids

    # Save
    output_path = eval_dir / "ground_truth.json"
    with open(output_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Generated ground truth for {len(ground_truth)} queries")
    print(f"Saved to {output_path}")

    # Summary
    print("\nSummary:")
    for qid, uids in list(ground_truth.items())[:5]:
        query_text = next(q["text"] for q in queries_data["queries"] if q["id"] == qid)
        print(f"  {qid}: '{query_text}' -> {len(uids)} relevant models")
    print("  ...")


if __name__ == "__main__":
    main()
