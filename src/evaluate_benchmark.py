"""
Large-scale Evaluation Script with Concurrent LLM Calls.

Runs the full EviRank pipeline (evidence mining + listwise re-ranking) on a
benchmark dataset with parallel processing for efficiency.

Supports:
- Concurrent API calls via ThreadPoolExecutor
- Incremental result saving (resume after interruption)
- Multiple ground-truth images per query (e.g., SoP dataset)
- Configurable EviRank variants (mini/plus/pro)

Usage:
    python -m src.evaluate_benchmark \
        --config configs/default.yaml \
        --dataset sop \
        --variant pro \
        --workers 32 \
        --output results/sop_evirank_pro.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Dict, List, Optional, Set

import numpy as np
from tqdm import tqdm

from src.evidence.evidence_frame import EvidenceFrame
from src.evidence.unified_evidence_generator import UnifiedEvidenceGenerator
from src.models.teacher_model import TeacherModel
from src.reranking.rubric_scorer import RubricScorer
from src.reranking.listwise_ranker import ListwiseRanker
from src.evaluation.metrics import evaluate_retrieval
from src.utils.config import load_config
from src.utils.privacy import make_opaque_id, safe_error_name


# ============================================================
# Metrics (per-query, supports multiple ground truths)
# ============================================================

def recall_at_k(gt_set: Set[str], ranked: List[str], k: int) -> float:
    """Recall@K with multiple ground truths."""
    if not gt_set:
        return 0.0
    return float(any(p in gt_set for p in ranked[:k]))


def mrr(gt_set: Set[str], ranked: List[str], max_rank: Optional[int] = None) -> float:
    """Mean Reciprocal Rank."""
    cutoff = ranked[:max_rank] if max_rank else ranked
    for i, item in enumerate(cutoff):
        if item in gt_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(gt_set: Set[str], ranked: List[str], k: int) -> float:
    """nDCG@K with binary relevance."""
    if not gt_set:
        return 0.0
    dcg = sum(
        1.0 / np.log2(i + 2) for i, p in enumerate(ranked[:k]) if p in gt_set
    )
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt_set), k)))
    return float(dcg / ideal_dcg) if ideal_dcg > 0 else 0.0


def compute_query_metrics(gt_paths: List[str], ranked: List[str]) -> Dict[str, float]:
    """Compute all metrics for a single query."""
    gt_set = set(gt_paths)
    return {
        "recall@1": recall_at_k(gt_set, ranked, 1),
        "recall@5": recall_at_k(gt_set, ranked, 5),
        "recall@10": recall_at_k(gt_set, ranked, 10),
        "ndcg@5": ndcg_at_k(gt_set, ranked, 5),
        "ndcg@10": ndcg_at_k(gt_set, ranked, 10),
        "mrr": mrr(gt_set, ranked),
        "mrr@5": mrr(gt_set, ranked, 5),
    }


# ============================================================
# Pipeline
# ============================================================

class EviRankEvaluator:
    """Concurrent EviRank evaluation on benchmark datasets."""

    def __init__(
        self,
        config: Dict,
        variant: str = "pro",
        max_workers: int = 32,
    ):
        self.config = config
        self.variant = variant
        self.max_workers = max_workers

        # Initialize teacher
        self.teacher = TeacherModel.from_config(config)

        # Initialize pipeline components
        self.evidence_generator = UnifiedEvidenceGenerator(self.teacher)
        self.rubric_scorer = RubricScorer()
        rerank_cfg = config.get("reranking", {})
        self.listwise_top_m = int(rerank_cfg.get("listwise_top_m", 5))
        batch_size = int(rerank_cfg.get("batch_size", 5))
        allow_divide_and_merge = bool(
            rerank_cfg.get("allow_divide_and_merge", False)
        )
        if self.listwise_top_m > batch_size and not allow_divide_and_merge:
            raise ValueError(
                "reranking.listwise_top_m must be <= reranking.batch_size "
                "when allow_divide_and_merge is false."
            )
        self.listwise_ranker = ListwiseRanker(
            self.teacher,
            batch_size=batch_size,
            allow_divide_and_merge=allow_divide_and_merge,
        )

        # Thread-safe state
        self._lock = Lock()
        self._results: List[Dict] = []

    def evaluate_query(
        self,
        query_caption: str,
        candidate_ids: List[str],
        candidate_captions: List[str],
        evidence_data: Optional[Dict] = None,
    ) -> List[str]:
        """
        Run EviRank on a single query and return re-ranked candidate IDs.

        Args:
            query_caption: Text description of the query.
            candidate_ids: Original-order candidate IDs.
            candidate_captions: Candidate descriptions/captions.
            evidence_data: Pre-computed evidence (skip mining if provided).

        Returns:
            Re-ranked candidate IDs.
        """
        # Build or load evidence frame
        if evidence_data:
            evidence_frame = EvidenceFrame.from_dict(evidence_data.get("b1_evidence", {}))
            b0 = evidence_data.get("b0_expansion", {})
            evidence_frame.core_intent = b0.get("core_intent", query_caption)
            evidence_frame.expanded_query = b0.get("expanded_query", query_caption)
            evidence_frame.confidence = b0.get("confidence", 0.9)
        else:
            # One fused P-CAP/P-B0/P-B1 request.
            evidence_output = self.evidence_generator.generate(
                query_text=query_caption,
                query_mode="text-only",
            )
            evidence_frame = evidence_output.evidence_frame

        # Rubric scoring (all variants use this)
        rubric_scores = []
        for cap in candidate_captions:
            score = self.rubric_scorer.score_candidate(cap, evidence_frame)
            rubric_scores.append(score["total_score"])

        # Sort by rubric score
        ranked_indices = sorted(
            range(len(candidate_ids)),
            key=lambda i: rubric_scores[i],
            reverse=True,
        )

        if self.variant == "mini":
            # EviRank-mini: rubric only
            return [candidate_ids[i] for i in ranked_indices]

        if self.variant in ("plus", "pro"):
            # EviRank-plus/pro: listwise refinement on top candidates
            top_n = min(self.listwise_top_m, len(candidate_ids))
            top_indices = ranked_indices[:top_n]
            top_candidates = [
                {"id": candidate_ids[i], "caption": candidate_captions[i]}
                for i in top_indices
            ]

            reranked, _ = self.listwise_ranker.rerank(
                query_image=None,
                query_caption=query_caption,
                evidence_frame=evidence_frame,
                candidates=top_candidates,
            )

            # Merge: reranked top + remaining in rubric order
            reranked_ids = [c["id"] for c in reranked]
            remaining = [candidate_ids[i] for i in ranked_indices[top_n:]]
            return reranked_ids + remaining

        # Default (student variant would go here)
        return [candidate_ids[i] for i in ranked_indices]

    def run_benchmark(
        self,
        queries: List[Dict],
        output_path: str,
    ) -> Dict[str, float]:
        """
        Run evaluation on a list of queries with concurrent processing.

        Each query dict should contain:
            - query_id / query_image: identifier
            - query_caption / generated_caption: text description
            - gt_paths: list of ground-truth IDs
            - candidates: list of candidate IDs
            - candidate_captions: list of candidate descriptions
            - evidence (optional): pre-computed evidence dict

        Returns:
            Aggregated metrics.
        """
        all_metrics: List[Dict[str, float]] = []
        save_lock = Lock()

        def process_one(query: Dict) -> Optional[Dict[str, float]]:
            try:
                caption = query.get("generated_caption") or query.get("query_caption", "")
                candidates = query.get("candidates", [])
                captions = query.get("candidate_captions", [""] * len(candidates))
                evidence = query.get("evidence")

                ranked = self.evaluate_query(caption, candidates, captions, evidence)

                gt = query.get("gt_paths", [])
                metrics = compute_query_metrics(gt, ranked)

                # Save incrementally
                with save_lock:
                    raw_query_id = (
                        query.get("query_id")
                        or query.get("image_id")
                        or query.get("reference_id")
                        or query.get("query_image", "")
                    )
                    record = {
                        "query_id": make_opaque_id(
                            raw_query_id,
                            namespace="query",
                        ),
                        "ranked": [
                            make_opaque_id(item_id, namespace="candidate")
                            for item_id in ranked
                        ],
                        "metrics": metrics,
                    }
                    self._results.append(record)

                    # Periodic save
                    if len(self._results) % 100 == 0:
                        self._save_partial(output_path)

                return metrics
            except Exception as e:
                print(f"Error processing query: {safe_error_name(e)}")
                return None

        # Concurrent execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_one, q): q for q in queries}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                result = future.result()
                if result:
                    all_metrics.append(result)

        # Final save
        self._save_partial(output_path)

        # Aggregate metrics
        if not all_metrics:
            return {}

        agg = {}
        keys = all_metrics[0].keys()
        for k in keys:
            agg[k] = float(np.mean([m[k] for m in all_metrics]))
        agg["num_queries"] = len(all_metrics)

        print(f"\n{'='*60}")
        print("Aggregated Results:")
        for k, v in sorted(agg.items()):
            print(f"  {k}: {v:.4f}")
        print(f"{'='*60}")

        return agg

    def _save_partial(self, output_path: str) -> None:
        """Save partial results for resume capability."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self._results, f, indent=2, ensure_ascii=False)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="EviRank Benchmark Evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--variant", default="pro", choices=["mini", "plus", "pro"])
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--output", default="results/benchmark_results.json")
    parser.add_argument("--evidence-path", default=None, help="Pre-computed evidence JSON")
    parser.add_argument("--test-path", default=None, help="Test data JSON")
    parser.add_argument("--sample-k", type=int, default=None, help="Sample K queries")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config = load_config(args.config)

    evaluator = EviRankEvaluator(
        config=config,
        variant=args.variant,
        max_workers=args.workers,
    )

    # Load test data
    if args.test_path:
        with open(args.test_path) as f:
            queries = json.load(f)
    else:
        print("Error: --test-path is required")
        return

    # Optional: load pre-computed evidence
    if args.evidence_path:
        with open(args.evidence_path) as f:
            evidence_list = json.load(f)
        # Build evidence map by opaque ID. Legacy raw identifiers are hashed
        # locally and never included in external requests or new artifacts.
        evidence_map = {}
        for item in evidence_list:
            key = item.get("opaque_id")
            if not key:
                legacy_key = item.get("path") or item.get("image_id", "")
                key = make_opaque_id(legacy_key, namespace="query")
            evidence_map[key] = item

        # Attach evidence to queries
        for q in queries:
            raw_qkey = (
                q.get("query_id")
                or q.get("image_id")
                or q.get("reference_id")
                or q.get("query_image", "")
            )
            qkey = make_opaque_id(raw_qkey, namespace="query")
            if qkey in evidence_map:
                q["evidence"] = evidence_map[qkey]

    # Optional sampling
    if args.sample_k and args.sample_k < len(queries):
        import random
        random.seed(args.seed)
        queries = random.sample(queries, args.sample_k)
        print(f"Sampled {args.sample_k} queries (seed={args.seed})")

    # Run evaluation
    results = evaluator.run_benchmark(queries, args.output)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
