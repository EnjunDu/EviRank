"""
P-RANK: Listwise Re-ranking with Evidence Constraints (Section I.4, Figure 8).

Performs evidence-conditioned listwise comparison using the MLLM teacher.
Supports divide-and-merge batching for large candidate sets (K > 8).

Also identifies hard-pair cases for distillation (Section 3.3).
"""

from __future__ import annotations

import base64
import json
import math
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from src.evidence.evidence_frame import EvidenceFrame
from src.models.teacher_model import TeacherModel
from src.utils.privacy import redact_external_text


# Prompt template from Figure 8 of the paper
PROMPT_P_RANK = """You are a visual reranker for image-to-image product retrieval. Given a query image, its detailed description, structured evidence constraints, and a list of candidate images, rank the candidate images from best to worst match based on visual similarity to the query image.

The goal is to find images of the SAME product or very similar products as the query image. Focus on:
- Exact or near-exact visual matches, including the same product from different angles.
- Products with identical key visual features, such as shape, color, branding, and distinctive details.
- Products from the same product line or category.

In addition to the final ranking, identify hard-pair cases: candidate pairs whose relevance scores are very close or visually difficult to distinguish. These hard pairs will be used as fine-grained distillation signals for training the student re-ranker.

IMPORTANT OUTPUT RULES:
- Output MUST be a single JSON object.
- JSON schema:
{{
  "ranked_indices": [int, int, ...],
  "hard_pairs": [
    {{"pair": [int, int], "reason": "short reason"}}
  ],
  "brief_reasoning": "one short paragraph explaining the ranking"
}}
- "ranked_indices" must be a permutation of 0 .. {n_minus_1}, ordered from BEST match to WORST match.
- "hard_pairs" should contain visually ambiguous candidate pairs with close relevance to the query.
- Do NOT output any extra text outside JSON.

=== QUERY IMAGE DESCRIPTION ===
{generated_caption}

=== EVIDENCE CONSTRAINTS ===
{evidence_text}

=== CANDIDATE LIST (index -> opaque ID) ===
{candidates_desc}

Below you will see:
1. The QUERY IMAGE (labeled) - this is what you are trying to match
2. The CANDIDATE IMAGES in order (0, 1, 2, ...) - rank these by similarity to the query"""


class ListwiseRanker:
    """
    Evidence-conditioned listwise re-ranking using MLLM teacher.

    Implements the P-RANK stage with divide-and-merge batching for
    large candidate sets (Section I.4).

    Supports multi-ground-truth datasets (e.g., SoP where the same product
    may appear from different viewpoints/angles).
    """

    def __init__(
        self,
        teacher_model: TeacherModel,
        batch_size: int = 5,
        max_image_edge: int = 2048,
        jpeg_quality: int = 85,
        image_detail: str = "low",
        allow_divide_and_merge: bool = False,
    ):
        """
        Args:
            teacher_model: MLLM teacher for listwise comparison.
            batch_size: Group size B for divide-and-merge (default 5).
            max_image_edge: Max edge length for image encoding.
            jpeg_quality: JPEG quality for base64 encoding.
            image_detail: Image transport detail level ("low"/"high"/"auto").
            allow_divide_and_merge: Permit additional teacher calls for
                candidate sets larger than ``batch_size``. Disabled by
                default so the standard top-M path makes exactly one
                listwise request.
        """
        self.teacher = teacher_model
        self.batch_size = batch_size
        self.max_image_edge = max_image_edge
        self.jpeg_quality = jpeg_quality
        self.image_detail = image_detail
        self.allow_divide_and_merge = allow_divide_and_merge

    def rerank(
        self,
        query_image: Optional[Image.Image],
        query_caption: str,
        evidence_frame: EvidenceFrame,
        candidates: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Re-rank candidates using evidence-conditioned listwise comparison.

        Args:
            query_image: The query image (PIL Image).
            query_caption: Caption/description of the query.
            evidence_frame: Structured evidence frame E(q).
            candidates: List of candidate dicts with 'id', 'image', 'caption'.

        Returns:
            Tuple of (reranked_candidates, hard_pairs).
        """
        n = len(candidates)

        if n == 0:
            return [], []

        if n <= self.batch_size:
            # Direct listwise ranking for small sets
            return self._rank_group(
                query_image, query_caption, evidence_frame, candidates
            )

        if not self.allow_divide_and_merge:
            raise ValueError(
                "Listwise candidate count exceeds batch_size. Select rubric "
                "top-M with M <= batch_size to preserve one listwise API call."
            )

        # Optional divide-and-merge for non-default large-set experiments.
        return self._divide_and_merge(
            query_image, query_caption, evidence_frame, candidates
        )

    def _rank_group(
        self,
        query_image: Optional[Image.Image],
        query_caption: str,
        evidence_frame: EvidenceFrame,
        candidates: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Rank a single group of candidates via teacher."""
        n = len(candidates)

        # Build evidence text (Section I.5)
        evidence_text = redact_external_text(evidence_frame.to_evidence_text())

        # Build candidate descriptions (anonymized indices)
        cand_desc_lines = []
        for i, cand in enumerate(candidates):
            cap = redact_external_text(
                cand.get("caption", "No description available")
            )
            cand_desc_lines.append(f"Candidate {i}: {cap}")
        candidates_desc = "\n".join(cand_desc_lines)

        # Format prompt
        prompt = PROMPT_P_RANK.format(
            n_minus_1=n - 1,
            generated_caption=redact_external_text(query_caption),
            evidence_text=evidence_text,
            candidates_desc=candidates_desc,
        )

        # Prepare images for multimodal call
        images = []
        if query_image is not None:
            images.append(query_image)
        for cand in candidates:
            img = cand.get("image")
            if img is not None:
                images.append(img)

        # Call teacher
        if images:
            response = self.teacher.generate_text_from_images(
                images=images,
                prompt=prompt,
            )
        else:
            response = self.teacher.generate_text(prompt=prompt)

        # Parse response
        ranked_indices, hard_pairs = self._parse_ranking_response(response, n)

        # Reorder candidates
        reranked = []
        for idx in ranked_indices:
            if 0 <= idx < n:
                reranked.append(candidates[idx])

        # Append any missing candidates in original order
        seen = set(ranked_indices)
        for i, cand in enumerate(candidates):
            if i not in seen:
                reranked.append(cand)

        return reranked, hard_pairs

    def _divide_and_merge(
        self,
        query_image: Optional[Image.Image],
        query_caption: str,
        evidence_frame: EvidenceFrame,
        candidates: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Divide-and-merge strategy for large candidate sets.

        Partition into groups of size B, rank each group independently,
        then iteratively merge via pairwise re-ranking calls.
        Requires ceil(K/B) + ceil(log2(ceil(K/B))) calls.
        """
        n = len(candidates)
        B = self.batch_size

        # Partition into groups
        groups = [candidates[i:i + B] for i in range(0, n, B)]
        all_hard_pairs = []

        # Rank each group independently
        ranked_groups = []
        for group in groups:
            ranked, hard = self._rank_group(
                query_image, query_caption, evidence_frame, group
            )
            ranked_groups.append(ranked)
            all_hard_pairs.extend(hard)

        # Iterative merging
        while len(ranked_groups) > 1:
            new_groups = []
            for i in range(0, len(ranked_groups), 2):
                if i + 1 < len(ranked_groups):
                    # Merge two groups via re-ranking
                    merged_candidates = ranked_groups[i] + ranked_groups[i + 1]
                    # Take top candidates for merge comparison
                    top_from_each = (
                        ranked_groups[i][:B] + ranked_groups[i + 1][:B]
                    )
                    merged, hard = self._rank_group(
                        query_image, query_caption, evidence_frame, top_from_each
                    )
                    # Append remaining candidates in original rank order
                    seen_ids = {c.get("id") for c in merged}
                    for c in merged_candidates:
                        if c.get("id") not in seen_ids:
                            merged.append(c)
                            seen_ids.add(c.get("id"))
                    new_groups.append(merged)
                    all_hard_pairs.extend(hard)
                else:
                    new_groups.append(ranked_groups[i])
            ranked_groups = new_groups

        return ranked_groups[0] if ranked_groups else [], all_hard_pairs

    def _parse_ranking_response(
        self,
        response: str,
        n: int,
    ) -> Tuple[List[int], List[Dict[str, Any]]]:
        """Parse the JSON ranking response from teacher."""
        hard_pairs = []

        # Try JSON parse
        data = self._extract_json(response)
        if data and "ranked_indices" in data:
            indices = data["ranked_indices"]
            if isinstance(indices, list):
                # Validate indices
                valid_indices = [int(i) for i in indices if 0 <= int(i) < n]
                if len(valid_indices) == n:
                    # Parse hard pairs
                    for hp in data.get("hard_pairs", []):
                        if isinstance(hp, dict) and "pair" in hp:
                            hard_pairs.append(hp)
                    return valid_indices, hard_pairs

        # Fallback: try to extract numbers from response
        numbers = re.findall(r'\b(\d+)\b', response)
        indices = []
        seen = set()
        for num_str in numbers:
            num = int(num_str)
            if 0 <= num < n and num not in seen:
                indices.append(num)
                seen.add(num)
            if len(indices) == n:
                break

        if not indices:
            indices = list(range(n))

        return indices, hard_pairs

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        """Extract JSON from response text."""
        if not text:
            return None
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        match = re.search(r'\{.*\}', text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None
