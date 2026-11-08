"""
Student Re-ranker Model (Section 3.3: Evidence as Supervision).

A lightweight cross-modal scoring network that learns from the teacher's
structured evidence and listwise ranking decisions.

Architecture:
- Text projection: Linear -> ReLU -> Dropout -> Linear
- Image projection: Linear -> ReLU -> Dropout -> Linear
- Cross-attention: MultiheadAttention(query, image)
- Scoring MLP: [query || image || caption || cross_attn] -> score

Training objectives (Eq. 7-10):
- L_rank: listwise ranking loss from teacher permutations
- L_slot: per-slot constraint satisfaction loss
- L_hard: hard-pair margin loss
- L_total = L_rank + lambda * L_slot + eta * L_hard
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class StudentRanker(nn.Module):
    """
    Lightweight student model for evidence-aware re-ranking.

    Distilled from MLLM teacher using structured evidence supervision.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Dimension of input embeddings (from backbone).
            hidden_dim: Hidden dimension for projections.
            num_heads: Number of attention heads.
            num_layers: Number of layers (unused, kept for compat).
            dropout: Dropout rate.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Query text projection
        self.query_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Candidate image projection
        self.image_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Candidate caption projection
        self.caption_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Cross-modal attention: query attends to image
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Scoring network: concatenation of [query, image, caption, cross_attn]
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        query_emb: torch.Tensor,
        image_emb: torch.Tensor,
        caption_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ranking scores for candidates.

        Args:
            query_emb: [1, D] or [B, D] query embeddings.
            image_emb: [K, D] candidate image embeddings.
            caption_emb: [K, D] candidate caption embeddings.

        Returns:
            scores: [K] relevance scores.
        """
        # Project
        q = self.query_proj(query_emb)  # [1, H]
        q = nn.functional.normalize(q, p=2, dim=-1)

        img = self.image_proj(image_emb)  # [K, H]
        img = nn.functional.normalize(img, p=2, dim=-1)

        cap = self.caption_proj(caption_emb)  # [K, H]
        cap = nn.functional.normalize(cap, p=2, dim=-1)

        K = image_emb.size(0)

        # Expand query for all candidates
        q_exp = q.expand(K, -1)  # [K, H]

        # Cross-attention: query attends to image (per candidate)
        q_attn_in = q_exp.unsqueeze(1)  # [K, 1, H]
        img_attn_in = img.unsqueeze(1)  # [K, 1, H]
        attn_out, _ = self.cross_attention(q_attn_in, img_attn_in, img_attn_in)
        attn_out = attn_out.squeeze(1)  # [K, H]

        # Concatenate features
        combined = torch.cat([q_exp, img, cap, attn_out], dim=-1)  # [K, 4H]

        # Score
        scores = self.scorer(combined).squeeze(-1)  # [K]
        return scores

    def rank_candidates(
        self,
        query_emb: np.ndarray,
        image_embs: np.ndarray,
        caption_embs: np.ndarray,
    ) -> List[Tuple[int, float]]:
        """
        Rank candidates by score (inference mode).

        Args:
            query_emb: [D] query embedding (numpy).
            image_embs: [K, D] candidate image embeddings (numpy).
            caption_embs: [K, D] candidate caption embeddings (numpy).

        Returns:
            List of (index, score) sorted by descending score.
        """
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            q = torch.from_numpy(query_emb).float().unsqueeze(0).to(device)
            img = torch.from_numpy(image_embs).float().to(device)
            cap = torch.from_numpy(caption_embs).float().to(device)
            scores = self.forward(q, img, cap)
            scores_np = scores.cpu().numpy()

        ranked = sorted(enumerate(scores_np), key=lambda x: x[1], reverse=True)
        return ranked

    def save_checkpoint(self, path: str, **kwargs) -> None:
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ckpt = {
            "model_state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
        }
        ckpt.update(kwargs)
        torch.save(ckpt, path)

    @classmethod
    def load_from_checkpoint(cls, path: str, device: str = "cuda") -> "StudentRanker":
        """Load model from checkpoint."""
        ckpt = torch.load(path, map_location=device)
        model = cls(
            input_dim=ckpt.get("input_dim", 2048),
            hidden_dim=ckpt.get("hidden_dim", 512),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        return model
