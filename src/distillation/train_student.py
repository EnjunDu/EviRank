"""
Student Distillation Training (Section 3.3: Evidence as Supervision).

Training objectives:
- L_rank: Listwise ranking loss from teacher permutations (Eq. 7)
- L_slot: Per-slot constraint satisfaction regression (Eq. 8)
- L_hard: Hard-pair margin loss for fine-grained disambiguation (Eq. 9)
- L_total = L_rank + lambda * L_slot + eta * L_hard (Eq. 10)

Hyperparameters (Section 4):
- lambda and eta selected via grid search on validation set
- Student: Qwen3-VL-2B-Thinking
- 20k queries (4K x 5 datasets)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.models.student_model import StudentRanker


def listwise_ranking_loss(
    scores: torch.Tensor,
    teacher_permutation: List[int],
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Listwise ranking loss (Eq. 7): KL divergence between student's
    score distribution and teacher's ranking.

    Args:
        scores: [K] predicted scores from student.
        teacher_permutation: Teacher's ranking (indices in best-to-worst order).
        temperature: Softmax temperature.

    Returns:
        Scalar loss.
    """
    K = len(teacher_permutation)
    # Create target distribution from teacher permutation
    # Higher rank -> higher target probability
    target_scores = torch.zeros(K, device=scores.device)
    for rank, idx in enumerate(teacher_permutation):
        target_scores[idx] = float(K - rank) / K

    # KL divergence
    log_pred = torch.log_softmax(scores / temperature, dim=0)
    target_dist = torch.softmax(target_scores / temperature, dim=0)
    loss = torch.sum(target_dist * (torch.log(target_dist + 1e-12) - log_pred))
    return loss


def slot_satisfaction_loss(
    scores: torch.Tensor,
    slot_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Per-slot constraint satisfaction loss (Eq. 8).

    Args:
        scores: [K] predicted scores.
        slot_labels: [K] per-candidate rubric satisfaction scores from teacher.

    Returns:
        Scalar MSE loss.
    """
    return nn.functional.mse_loss(torch.sigmoid(scores), slot_labels)


def hard_pair_margin_loss(
    scores: torch.Tensor,
    hard_pairs: List[Tuple[int, int]],
    margin: float = 0.1,
) -> torch.Tensor:
    """
    Hard-pair margin loss (Eq. 9): enforce score margin between
    hard-to-distinguish pairs identified by teacher.

    Args:
        scores: [K] predicted scores.
        hard_pairs: List of (better_idx, worse_idx) pairs from teacher.
        margin: Minimum score margin.

    Returns:
        Scalar loss.
    """
    if not hard_pairs:
        return torch.tensor(0.0, device=scores.device)

    loss = torch.tensor(0.0, device=scores.device)
    for better, worse in hard_pairs:
        diff = scores[better] - scores[worse]
        loss += torch.relu(margin - diff)

    return loss / max(1, len(hard_pairs))


def train_student(
    student_model: StudentRanker,
    train_data: List[Dict],
    config: Dict,
    val_data: Optional[List[Dict]] = None,
) -> str:
    """
    Train the student re-ranker via knowledge distillation.

    Args:
        student_model: The student model to train.
        train_data: Training examples, each containing:
            - query_emb: [D] query embedding
            - image_embs: [K, D] candidate image embeddings
            - caption_embs: [K, D] candidate caption embeddings
            - teacher_permutation: list of indices (best to worst)
            - slot_labels: [K] rubric scores
            - hard_pairs: list of (better, worse) tuples
            - confidence: float (from P-B0)
        config: Training configuration dict.
        val_data: Optional validation data.

    Returns:
        Path to saved checkpoint.
    """
    training_cfg = config.get("training", {})
    distill_cfg = config.get("distillation", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model.to(device)
    student_model.train()

    # Hyperparameters
    num_epochs = int(training_cfg.get("num_epochs", 10))
    lr = float(training_cfg.get("learning_rate", 1e-4))
    weight_decay = float(training_cfg.get("weight_decay", 0.01))
    grad_clip = float(training_cfg.get("gradient_clip", 1.0))
    lambda_slot = float(distill_cfg.get("lambda_slot", 0.5))
    eta_hard = float(distill_cfg.get("eta_hard", 0.3))
    temperature = float(distill_cfg.get("temperature", 3.0))
    output_dir = str(training_cfg.get("output_dir", "checkpoints"))

    optimizer = optim.AdamW(
        student_model.parameters(), lr=lr, weight_decay=weight_decay
    )

    best_loss = float("inf")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_data, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for sample in pbar:
            query_emb = torch.from_numpy(
                np.asarray(sample["query_emb"])
            ).float().unsqueeze(0).to(device)
            image_embs = torch.from_numpy(
                np.asarray(sample["image_embs"])
            ).float().to(device)
            caption_embs = torch.from_numpy(
                np.asarray(sample["caption_embs"])
            ).float().to(device)

            # Forward
            scores = student_model(query_emb, image_embs, caption_embs)

            # Compute losses (Eq. 10)
            l_rank = listwise_ranking_loss(
                scores, sample["teacher_permutation"], temperature
            )

            slot_labels = torch.tensor(
                sample.get("slot_labels", [0.5] * len(scores)),
                device=device,
            ).float()
            l_slot = slot_satisfaction_loss(scores, slot_labels)

            hard_pairs = sample.get("hard_pairs", [])
            l_hard = hard_pair_margin_loss(scores, hard_pairs)

            # Weighted by confidence (Eq. 10)
            conf = float(sample.get("confidence", 1.0))
            loss = conf * (l_rank + lambda_slot * l_slot + eta_hard * l_hard)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student_model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / max(1, len(train_data))
        print(f"Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    # Save checkpoint
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, "student_ranker.pt")
    student_model.save_checkpoint(
        ckpt_path,
        epoch=num_epochs,
        best_loss=best_loss,
    )
    print(f"Saved checkpoint: {ckpt_path}")
    return ckpt_path
