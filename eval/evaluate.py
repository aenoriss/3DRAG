"""
3DRAG Retrieval Evaluation Script

Evaluates semantic search quality using:
- MRR (Mean Reciprocal Rank)
- Recall@k
- NDCG@k (Normalized Discounted Cumulative Gain)
- Per-category breakdown

Usage:
    python eval/evaluate.py --api http://localhost:8000
    python eval/evaluate.py --api http://localhost:8000 --k 10
"""

import json
import argparse
import requests
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class QueryResult:
    query_id: str
    query_text: str
    category: str
    retrieved: List[str]  # UIDs in rank order
    relevant: List[str]   # Ground truth UIDs
    relevance_scores: Optional[Dict[str, int]] = None  # UID -> score (1-3)


def load_queries(path: Path) -> List[dict]:
    """Load test queries from JSON."""
    with open(path) as f:
        data = json.load(f)
    return data["queries"]


def load_ground_truth(path: Path) -> Dict[str, List[str]]:
    """Load ground truth mappings (query_id -> relevant UIDs)."""
    with open(path) as f:
        return json.load(f)


def search(api_url: str, query: str, k: int = 10) -> List[str]:
    """Query the search API and return UIDs."""
    response = requests.get(
        f"{api_url}/search",
        params={"q": query, "limit": k}
    )
    response.raise_for_status()
    results = response.json()
    return [r["uid"] for r in results.get("results", [])]


# === Metrics ===

def reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    """Compute reciprocal rank (1/rank of first relevant item)."""
    for i, uid in enumerate(retrieved):
        if uid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Compute Recall@k."""
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_k & relevant_set) / len(relevant_set)


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Compute Precision@k."""
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    relevant_set = set(relevant)
    return sum(1 for uid in retrieved_k if uid in relevant_set) / len(retrieved_k)


def dcg_at_k(retrieved: List[str], relevance_scores: Dict[str, int], k: int) -> float:
    """Compute DCG@k with graded relevance."""
    dcg = 0.0
    for i, uid in enumerate(retrieved[:k]):
        rel = relevance_scores.get(uid, 0)
        dcg += (2**rel - 1) / np.log2(i + 2)  # i+2 because log2(1) = 0
    return dcg


def ndcg_at_k(retrieved: List[str], relevance_scores: Dict[str, int], k: int) -> float:
    """Compute NDCG@k (normalized DCG)."""
    dcg = dcg_at_k(retrieved, relevance_scores, k)

    # Ideal DCG: sort by relevance scores
    ideal_order = sorted(relevance_scores.keys(), key=lambda x: -relevance_scores[x])
    idcg = dcg_at_k(ideal_order, relevance_scores, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def binary_ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Compute NDCG@k with binary relevance (relevant=1, not relevant=0)."""
    relevance_scores = {uid: 1 for uid in relevant}
    return ndcg_at_k(retrieved, relevance_scores, k)


# === Evaluation ===

def evaluate(
    queries: List[dict],
    ground_truth: Dict[str, List[str]],
    api_url: str,
    k: int = 10
) -> dict:
    """Run full evaluation and compute metrics."""

    results = []

    print(f"\nRunning {len(queries)} queries against {api_url}...")
    print("-" * 50)

    for i, query in enumerate(queries):
        qid = query["id"]
        text = query["text"]
        category = query.get("category", "unknown")

        # Get ground truth
        relevant = ground_truth.get(qid, [])
        if not relevant:
            print(f"  Warning: No ground truth for query '{qid}'")
            continue

        # Search
        try:
            retrieved = search(api_url, text, k)
        except Exception as e:
            print(f"  Error on query '{qid}': {e}")
            continue

        results.append(QueryResult(
            query_id=qid,
            query_text=text,
            category=category,
            retrieved=retrieved,
            relevant=relevant
        ))

        # Progress
        rr = reciprocal_rank(retrieved, relevant)
        r5 = recall_at_k(retrieved, relevant, 5)
        print(f"  [{i+1}/{len(queries)}] {text[:40]:40} RR={rr:.2f} R@5={r5:.2f}")

    # Aggregate metrics
    metrics = compute_aggregate_metrics(results, k)

    return metrics


def compute_aggregate_metrics(results: List[QueryResult], k: int = 10) -> dict:
    """Compute aggregate metrics from all query results."""

    if not results:
        return {"error": "No results"}

    # Overall metrics
    mrr_values = [reciprocal_rank(r.retrieved, r.relevant) for r in results]
    recall_1 = [recall_at_k(r.retrieved, r.relevant, 1) for r in results]
    recall_5 = [recall_at_k(r.retrieved, r.relevant, 5) for r in results]
    recall_10 = [recall_at_k(r.retrieved, r.relevant, 10) for r in results]
    ndcg_10 = [binary_ndcg_at_k(r.retrieved, r.relevant, 10) for r in results]
    precision_5 = [precision_at_k(r.retrieved, r.relevant, 5) for r in results]

    overall = {
        "num_queries": len(results),
        "MRR": float(np.mean(mrr_values)),
        "Recall@1": float(np.mean(recall_1)),
        "Recall@5": float(np.mean(recall_5)),
        "Recall@10": float(np.mean(recall_10)),
        "NDCG@10": float(np.mean(ndcg_10)),
        "Precision@5": float(np.mean(precision_5)),
    }

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    per_category = {}
    for cat, cat_results in categories.items():
        mrr_cat = [reciprocal_rank(r.retrieved, r.relevant) for r in cat_results]
        recall_5_cat = [recall_at_k(r.retrieved, r.relevant, 5) for r in cat_results]
        per_category[cat] = {
            "num_queries": len(cat_results),
            "MRR": float(np.mean(mrr_cat)),
            "Recall@5": float(np.mean(recall_5_cat)),
        }

    # Failure analysis
    failures = [r for r in results if reciprocal_rank(r.retrieved, r.relevant) == 0]

    return {
        "overall": overall,
        "per_category": per_category,
        "failures": [
            {"query": f.query_text, "category": f.category}
            for f in failures[:10]  # Top 10 failures
        ],
        "num_failures": len(failures),
    }


def print_report(metrics: dict):
    """Print formatted evaluation report."""

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    overall = metrics["overall"]
    print(f"\nOverall Metrics ({overall['num_queries']} queries):")
    print("-" * 40)
    print(f"  MRR:         {overall['MRR']:.3f}")
    print(f"  Recall@1:    {overall['Recall@1']:.3f}")
    print(f"  Recall@5:    {overall['Recall@5']:.3f}")
    print(f"  Recall@10:   {overall['Recall@10']:.3f}")
    print(f"  NDCG@10:     {overall['NDCG@10']:.3f}")
    print(f"  Precision@5: {overall['Precision@5']:.3f}")

    # Quality assessment
    mrr = overall['MRR']
    if mrr >= 0.7:
        quality = "Excellent"
    elif mrr >= 0.5:
        quality = "Good"
    elif mrr >= 0.3:
        quality = "Acceptable"
    else:
        quality = "Needs Improvement"
    print(f"\n  Quality: {quality}")

    # Per-category
    print(f"\nPer-Category Breakdown:")
    print("-" * 40)
    for cat, cat_metrics in sorted(metrics["per_category"].items()):
        print(f"  {cat:30} MRR={cat_metrics['MRR']:.2f} R@5={cat_metrics['Recall@5']:.2f}")

    # Failures
    if metrics["num_failures"] > 0:
        print(f"\nFailure Analysis ({metrics['num_failures']} queries with no hits):")
        print("-" * 40)
        for f in metrics["failures"]:
            print(f"  - [{f['category']}] {f['query']}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3DRAG retrieval")
    parser.add_argument("--api", default="http://localhost:8000", help="API URL")
    parser.add_argument("--k", type=int, default=10, help="Top-k results to evaluate")
    parser.add_argument("--queries", default="eval/queries.json", help="Queries file")
    parser.add_argument("--ground-truth", default="eval/ground_truth.json", help="Ground truth file")
    parser.add_argument("--output", help="Save results to JSON file")
    args = parser.parse_args()

    # Load data
    queries_path = Path(args.queries)
    gt_path = Path(args.ground_truth)

    if not queries_path.exists():
        print(f"Error: Queries file not found: {queries_path}")
        print("Create eval/queries.json first. See EVALUATION_PLAN.md")
        return

    if not gt_path.exists():
        print(f"Error: Ground truth file not found: {gt_path}")
        print("Create eval/ground_truth.json first. See EVALUATION_PLAN.md")
        return

    queries = load_queries(queries_path)
    ground_truth = load_ground_truth(gt_path)

    # Run evaluation
    metrics = evaluate(queries, ground_truth, args.api, args.k)

    # Print report
    print_report(metrics)

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
