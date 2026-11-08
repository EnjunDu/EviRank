"""
Evaluation Metrics for Image Retrieval (Section 4).

Metrics reported in the paper:
- Recall@K (K = 1, 5, 10): fraction of queries where the ground-truth
  image appears in the top-K results.
- MRR@K: Mean Reciprocal Rank at K.
- MAP@K: Mean Average Precision at K.
- nDCG@K: Normalized Discounted Cumulative Gain at K.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def calculate_recall_at_k(
    ranked_results: List[List[str]],
    ground_truth: Dict[str, List[str]],
    k: int,
) -> float:
    """
    Recall@K: fraction of queries with ground-truth in top-K.

    Args:
        ranked_results: For each query, list of ranked candidate IDs.
        ground_truth: query_id -> list of relevant IDs.
        k: Cutoff.

    Returns:
        Recall@K score.
    """
    hits = 0
    total = 0

    for query_idx, ranked_list in enumerate(ranked_results):
        relevant = ground_truth.get(str(query_idx), [])
        if not relevant:
            continue
        total += 1
        if any(doc_id in relevant for doc_id in ranked_list[:k]):
            hits += 1

    return hits / max(total, 1)


def calculate_mrr(
    ranked_results: List[List[str]],
    ground_truth: Dict[str, List[str]],
    k: Optional[int] = None,
) -> float:
    """
    Mean Reciprocal Rank (optionally at K).

    Args:
        ranked_results: For each query, list of ranked candidate IDs.
        ground_truth: query_id -> list of relevant IDs.
        k: Optional cutoff (if None, consider full list).

    Returns:
        MRR score.
    """
    rr_sum = 0.0
    total = 0

    for query_idx, ranked_list in enumerate(ranked_results):
        relevant = ground_truth.get(str(query_idx), [])
        if not relevant:
            continue
        total += 1

        cutoff = ranked_list[:k] if k else ranked_list
        for i, doc_id in enumerate(cutoff):
            if doc_id in relevant:
                rr_sum += 1.0 / (i + 1)
                break

    return rr_sum / max(total, 1)


def calculate_map_at_k(
    ranked_results: List[List[str]],
    ground_truth: Dict[str, List[str]],
    k: int,
) -> float:
    """
    Mean Average Precision at K.

    Args:
        ranked_results: For each query, list of ranked candidate IDs.
        ground_truth: query_id -> list of relevant IDs.
        k: Cutoff.

    Returns:
        MAP@K score.
    """
    ap_scores = []

    for query_idx, ranked_list in enumerate(ranked_results):
        relevant = set(ground_truth.get(str(query_idx), []))
        if not relevant:
            continue

        num_relevant = 0
        precisions = []

        for i, doc_id in enumerate(ranked_list[:k]):
            if doc_id in relevant:
                num_relevant += 1
                precisions.append(num_relevant / (i + 1))

        ap = sum(precisions) / min(len(relevant), k) if precisions else 0.0
        ap_scores.append(ap)

    return float(np.mean(ap_scores)) if ap_scores else 0.0


def calculate_ndcg_at_k(
    ranked_results: List[List[str]],
    ground_truth: Dict[str, List[str]],
    k: int,
) -> float:
    """
    Normalized Discounted Cumulative Gain at K.

    Args:
        ranked_results: For each query, list of ranked candidate IDs.
        ground_truth: query_id -> list of relevant IDs.
        k: Cutoff.

    Returns:
        nDCG@K score.
    """
    ndcg_scores = []

    for query_idx, ranked_list in enumerate(ranked_results):
        relevant = set(ground_truth.get(str(query_idx), []))
        if not relevant:
            continue

        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(ranked_list[:k]):
            if doc_id in relevant:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because rank starts at 1

        # Ideal DCG
        ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))

        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def evaluate_retrieval(
    ranked_results: List[List[str]],
    ground_truth: Dict[str, List[str]],
    k_values: List[int] = None,
) -> Dict[str, float]:
    """
    Comprehensive retrieval evaluation.

    Args:
        ranked_results: For each query, list of ranked candidate IDs.
        ground_truth: query_id -> list of relevant IDs.
        k_values: List of K values for metrics.

    Returns:
        Dictionary with all metric scores.
    """
    if k_values is None:
        k_values = [1, 5, 10, 20, 50]

    results = {}

    for k in k_values:
        results[f"recall@{k}"] = calculate_recall_at_k(ranked_results, ground_truth, k)
        results[f"map@{k}"] = calculate_map_at_k(ranked_results, ground_truth, k)
        results[f"ndcg@{k}"] = calculate_ndcg_at_k(ranked_results, ground_truth, k)
        results[f"mrr@{k}"] = calculate_mrr(ranked_results, ground_truth, k)

    results["mrr"] = calculate_mrr(ranked_results, ground_truth)
    results["num_queries"] = len(ranked_results)

    return results
