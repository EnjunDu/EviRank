"""
Main Entry Point for EviRank Pipeline.

Provides CLI for running the full EviRank pipeline:
  1. preprocess  - Build FAISS index from embeddings
  2. retrieve    - Coarse retrieval + EviRank re-ranking
  3. evaluate    - Evaluate retrieval performance
  4. train       - Train student model via distillation
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict

from src.utils.config import load_config


def run_pipeline(config: Dict, query: str = None) -> None:
    """
    Run the full EviRank pipeline on a query.

    Pipeline stages:
    1. Coarse retrieval (FAISS top-K)
    2. Fused P-CAP/P-B0/P-B1 evidence request
    3. Rubric scoring over top-K (deterministic, local)
    4. P-RANK over rubric top-M (one teacher request)
    """
    from src.models.teacher_model import TeacherModel
    from src.evidence.unified_evidence_generator import UnifiedEvidenceGenerator
    from src.reranking.rubric_scorer import RubricScorer
    from src.reranking.listwise_ranker import ListwiseRanker
    from src.retrieval.coarse_retrieval import FAISSRetriever
    from src.utils.privacy import make_opaque_id

    # Initialize components
    teacher = TeacherModel.from_config(config)
    evidence_generator = UnifiedEvidenceGenerator(teacher)
    rubric_scorer = RubricScorer()
    rerank_cfg = config.get("reranking", {})
    batch_size = int(rerank_cfg.get("batch_size", 5))
    listwise_top_m = int(rerank_cfg.get("listwise_top_m", 5))
    allow_divide_and_merge = bool(
        rerank_cfg.get("allow_divide_and_merge", False)
    )
    if listwise_top_m > batch_size and not allow_divide_and_merge:
        raise ValueError(
            "reranking.listwise_top_m must be <= reranking.batch_size "
            "when allow_divide_and_merge is false."
        )
    listwise_ranker = ListwiseRanker(
        teacher,
        batch_size=batch_size,
        allow_divide_and_merge=allow_divide_and_merge,
    )

    # Coarse retrieval
    index_cfg = config.get("index", {})
    retriever = FAISSRetriever(
        index_path=index_cfg["index_path"],
        metadata_path=index_cfg["metadata_path"],
        top_k=int(config.get("retrieval", {}).get("top_k", 20)),
    )

    # Stage 1: Retrieve candidates
    candidates = retriever.search(query)
    print(f"[EviRank] Retrieved {len(candidates)} candidates")

    # Stage 2: one fused evidence request.
    evidence_output = evidence_generator.generate(
        query_text=query,
        query_mode="text-only",
    )
    evidence_frame = evidence_output.evidence_frame

    # Stage 4: Rubric scoring
    scored = []
    for cand in candidates:
        score = rubric_scorer.score_candidate(
            candidate_description=cand.get("caption", ""),
            evidence_frame=evidence_frame,
        )
        cand["rubric_score"] = score["total_score"]
        scored.append(cand)

    # Sort by rubric score
    scored.sort(key=lambda x: x["rubric_score"], reverse=True)

    # Stage 5: P-RANK - Listwise re-ranking (optional, teacher-based)
    use_listwise = config.get("reranking", {}).get("use_listwise", True)
    if use_listwise:
        top_candidates = scored[:listwise_top_m]
        reranked, hard_pairs = listwise_ranker.rerank(
            query_image=None,
            query_caption=evidence_output.generated_caption or query,
            evidence_frame=evidence_frame,
            candidates=top_candidates,
        )
        scored = reranked + scored[listwise_top_m:]

    # Display results
    show_n = int(config.get("retrieval", {}).get("show_top_n", 10))
    print(f"\n{'='*60}")
    print(f"Top-{show_n} Results:")
    print(f"{'='*60}")
    for i, cand in enumerate(scored[:show_n]):
        print(
            f"[{i+1:02d}] id={make_opaque_id(cand.get('id'), 'candidate')} "
            f"rubric={cand.get('rubric_score', 0):.4f} "
            f"faiss={cand.get('score', 0):.4f}"
        )


def run_evaluate(config: Dict, split: str = "test") -> None:
    """Run evaluation on a dataset split."""
    from src.evaluation.metrics import evaluate_retrieval

    print(f"[EviRank] Evaluating on {split} split...")
    # Implementation depends on dataset format
    # See src/evaluation/metrics.py for metric computation


def run_train(config: Dict) -> None:
    """Train student model via distillation."""
    from src.models.student_model import StudentRanker
    from src.distillation.train_student import train_student

    input_dim = int(config.get("index", {}).get("dimension", 2048))
    hidden_dim = int(config.get("training", {}).get("hidden_dim", 512))

    student = StudentRanker(input_dim=input_dim, hidden_dim=hidden_dim)
    # Note: train_data should be prepared from teacher's evidence mining output
    print("[EviRank] Training requires pre-generated distillation data.")
    print("[EviRank] Run evidence mining on training set first.")


def main():
    parser = argparse.ArgumentParser(
        description="EviRank: Structured Relevance Evidence for Multimodal Image Re-ranking"
    )
    parser.add_argument(
        "--mode",
        choices=["retrieve", "evaluate", "train", "preprocess"],
        required=True,
        help="Pipeline mode to run.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to configuration YAML file.",
    )
    parser.add_argument("--query", default=None, help="Query text for retrieve mode.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])

    args = parser.parse_args()
    config = load_config(args.config)

    if args.mode == "retrieve":
        if not args.query:
            print("Error: --query is required for retrieve mode.")
            sys.exit(1)
        run_pipeline(config, query=args.query)

    elif args.mode == "evaluate":
        run_evaluate(config, split=args.split)

    elif args.mode == "train":
        run_train(config)

    elif args.mode == "preprocess":
        from src.data.preprocess import build_faiss_index_from_embeddings
        print("[EviRank] Preprocessing requires pre-computed embeddings.")
        print("[EviRank] See scripts/preprocess.sh for usage.")


if __name__ == "__main__":
    main()
