"""
Deterministic Rubric Scoring (Section 3.2, Eq. 5).

Computes a slot-wise satisfaction score by matching required/forbidden
constraints against candidate image descriptions using text similarity.

Score formula (Eq. 5):
    S_rubric(c) = sum_{s in slots} w_s * [
        sum_{r in required_s} sim(r, desc_c) -
        sum_{f in forbidden_s} sim(f, desc_c)
    ]

where sim() is cosine similarity in embedding space.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

from src.evidence.evidence_frame import (
    ConstraintLabel,
    EvidenceFrame,
    SlotType,
)


class RubricScorer:
    """
    Deterministic rubric scoring based on structured evidence.

    This scorer computes how well each candidate satisfies or violates
    the evidence constraints, producing a transparent, per-slot breakdown.
    """

    # Default slot weights (can be adjusted via config)
    DEFAULT_WEIGHTS: Dict[SlotType, float] = {
        SlotType.ENTITIES: 0.25,
        SlotType.ATTRIBUTES: 0.20,
        SlotType.ACTIONS: 0.15,
        SlotType.RELATIONS: 0.15,
        SlotType.SCENE: 0.10,
        SlotType.KEY_DETAILS: 0.15,
    }

    def __init__(
        self,
        embedding_model=None,
        slot_weights: Optional[Dict[SlotType, float]] = None,
        similarity_threshold: float = 0.5,
    ):
        """
        Args:
            embedding_model: Model with encode_text() method for computing
                text embeddings. If None, falls back to keyword matching.
            slot_weights: Per-slot scoring weights. Defaults to paper values.
            similarity_threshold: Threshold for considering a match.
        """
        self.embedding_model = embedding_model
        self.weights = slot_weights or self.DEFAULT_WEIGHTS
        self.similarity_threshold = similarity_threshold

    def score_candidate(
        self,
        candidate_description: str,
        evidence_frame: EvidenceFrame,
    ) -> Dict[str, float]:
        """
        Score a single candidate against the evidence frame.

        Args:
            candidate_description: Text description/caption of the candidate.
            evidence_frame: The structured evidence frame E(q).

        Returns:
            Dictionary with per-slot scores and total_score.
        """
        scores = {}
        total = 0.0

        for slot_type in SlotType:
            slot = evidence_frame.get_slot(slot_type)
            weight = self.weights.get(slot_type, 0.1)

            # Compute required satisfaction
            req_score = self._compute_satisfaction(
                candidate_description,
                [c.text for c in slot.required],
            )

            # Compute forbidden violation (penalty)
            forb_score = self._compute_satisfaction(
                candidate_description,
                [c.text for c in slot.forbidden],
            )

            # Slot score = required_satisfaction - forbidden_violation (Eq. 5)
            slot_score = req_score - forb_score
            scores[f"{slot_type.value}_score"] = slot_score
            total += weight * slot_score

        scores["total_score"] = total
        return scores

    def score_candidates_batch(
        self,
        candidate_descriptions: List[str],
        evidence_frame: EvidenceFrame,
    ) -> List[Dict[str, float]]:
        """Score multiple candidates at once (more efficient with embeddings)."""
        if self.embedding_model is not None:
            return self._score_batch_with_embeddings(
                candidate_descriptions, evidence_frame
            )
        return [
            self.score_candidate(desc, evidence_frame)
            for desc in candidate_descriptions
        ]

    def _compute_satisfaction(
        self,
        candidate_text: str,
        constraint_texts: List[str],
    ) -> float:
        """
        Compute average satisfaction of constraints against candidate text.

        If embedding model is available, uses cosine similarity.
        Otherwise, falls back to keyword overlap.
        """
        if not constraint_texts:
            return 0.0

        if self.embedding_model is not None:
            return self._embedding_similarity(candidate_text, constraint_texts)

        return self._keyword_similarity(candidate_text, constraint_texts)

    def _embedding_similarity(
        self,
        candidate_text: str,
        constraint_texts: List[str],
    ) -> float:
        """Compute average cosine similarity using embeddings."""
        try:
            cand_emb = self.embedding_model.encode_text(candidate_text)
            const_embs = self.embedding_model.encode_text(constraint_texts)

            if cand_emb.ndim == 1:
                cand_emb = cand_emb[np.newaxis, :]

            # Normalize
            cand_norm = cand_emb / (np.linalg.norm(cand_emb, axis=-1, keepdims=True) + 1e-12)
            const_norm = const_embs / (np.linalg.norm(const_embs, axis=-1, keepdims=True) + 1e-12)

            # Cosine similarities
            sims = (const_norm @ cand_norm.T).squeeze(-1)
            return float(np.mean(sims))
        except Exception:
            return self._keyword_similarity(candidate_text, constraint_texts)

    def _keyword_similarity(
        self,
        candidate_text: str,
        constraint_texts: List[str],
    ) -> float:
        """Fallback: keyword-based overlap similarity."""
        if not constraint_texts:
            return 0.0

        cand_lower = candidate_text.lower()
        cand_words = set(cand_lower.split())
        total_sim = 0.0

        for constraint in constraint_texts:
            const_words = set(constraint.lower().split())
            if not const_words:
                continue
            overlap = len(cand_words & const_words) / len(const_words)
            total_sim += overlap

        return total_sim / len(constraint_texts)

    def _score_batch_with_embeddings(
        self,
        candidate_descriptions: List[str],
        evidence_frame: EvidenceFrame,
    ) -> List[Dict[str, float]]:
        """Batch scoring using embedding model for efficiency."""
        # Pre-encode all candidate descriptions
        try:
            all_cand_embs = self.embedding_model.encode_text(candidate_descriptions)
            if all_cand_embs.ndim == 1:
                all_cand_embs = all_cand_embs[np.newaxis, :]
            cand_norms = all_cand_embs / (
                np.linalg.norm(all_cand_embs, axis=-1, keepdims=True) + 1e-12
            )
        except Exception:
            return [
                self.score_candidate(desc, evidence_frame)
                for desc in candidate_descriptions
            ]

        results = []
        for i in range(len(candidate_descriptions)):
            scores = {}
            total = 0.0
            cand_vec = cand_norms[i:i+1]  # [1, D]

            for slot_type in SlotType:
                slot = evidence_frame.get_slot(slot_type)
                weight = self.weights.get(slot_type, 0.1)

                req_score = self._batch_sim(cand_vec, [c.text for c in slot.required])
                forb_score = self._batch_sim(cand_vec, [c.text for c in slot.forbidden])

                slot_score = req_score - forb_score
                scores[f"{slot_type.value}_score"] = slot_score
                total += weight * slot_score

            scores["total_score"] = total
            results.append(scores)

        return results

    def _batch_sim(self, cand_vec: np.ndarray, texts: List[str]) -> float:
        """Compute mean similarity between a candidate vector and constraint texts."""
        if not texts:
            return 0.0
        try:
            text_embs = self.embedding_model.encode_text(texts)
            if text_embs.ndim == 1:
                text_embs = text_embs[np.newaxis, :]
            text_norms = text_embs / (np.linalg.norm(text_embs, axis=-1, keepdims=True) + 1e-12)
            sims = (text_norms @ cand_vec.T).squeeze(-1)
            return float(np.mean(sims))
        except Exception:
            return 0.0
