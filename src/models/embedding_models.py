"""
Embedding Models for coarse retrieval.

Supports:
- CLIP (ViT-B/32, ViT-L/14)
- EVA-CLIP-18B
- BLIP-2
- DINOv2 (for image-to-image retrieval)
- Jina Embeddings v4

These backbones produce the initial top-K candidates that are then
re-ranked by EviRank (Section 4: Implementation Details).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return getattr(self, "embedding_dim", 512)

    @abstractmethod
    def encode_text(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode text(s) into embedding vector(s)."""
        ...

    @abstractmethod
    def encode_image(
        self,
        images: Union[Image.Image, List[Image.Image]],
        batch_size: int = 16,
    ) -> np.ndarray:
        """Encode image(s) into embedding vector(s)."""
        ...


class CLIPEmbeddingModel(EmbeddingModel):
    """
    OpenAI CLIP model (ViT-B/32 or ViT-L/14).

    Used as coarse retrieval backbone in the paper.
    """

    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        self.device = device
        self.model_name = model_name

        try:
            import clip
        except ImportError:
            raise ImportError(
                "The `clip` package is required. "
                "Install: pip install git+https://github.com/openai/CLIP.git"
            )

        self._clip = clip
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

        # Set embedding dimension based on model
        self.embedding_dim = 512 if "B/32" in model_name else 768

    def encode_text(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                tokens = self._clip.tokenize(batch, truncate=True).to(self.device)
                feats = self.model.encode_text(tokens)
                feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
                all_embs.append(feats.cpu().numpy())

        return np.vstack(all_embs).astype(np.float32)

    def encode_image(
        self,
        images: Union[Image.Image, List[Image.Image]],
        batch_size: int = 32,
    ) -> np.ndarray:
        if isinstance(images, Image.Image):
            images = [images]

        all_embs = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                tensors = torch.stack(
                    [self.preprocess(img).to(self.device) for img in batch]
                )
                feats = self.model.encode_image(tensors)
                feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
                all_embs.append(feats.cpu().numpy())

        return np.vstack(all_embs).astype(np.float32)


class EVACLIPEmbeddingModel(EmbeddingModel):
    """
    EVA-CLIP-18B embedding model.

    The strongest retrieval backbone in the paper (Table 1).
    Uses HuggingFace Transformers for loading.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.float16
        ).to(device)
        self.model.eval()
        self.embedding_dim = 2048

    def encode_text(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self.processor(text=batch, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model.get_text_features(**inputs)
                outputs = outputs / (outputs.norm(dim=-1, keepdim=True) + 1e-12)
                all_embs.append(outputs.cpu().float().numpy())

        return np.vstack(all_embs).astype(np.float32)

    def encode_image(
        self,
        images: Union[Image.Image, List[Image.Image]],
        batch_size: int = 8,
    ) -> np.ndarray:
        if isinstance(images, Image.Image):
            images = [images]

        all_embs = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.model.get_image_features(**inputs)
                outputs = outputs / (outputs.norm(dim=-1, keepdim=True) + 1e-12)
                all_embs.append(outputs.cpu().float().numpy())

        return np.vstack(all_embs).astype(np.float32)


class BLIP2EmbeddingModel(EmbeddingModel):
    """BLIP-2 embedding model for multimodal retrieval."""

    def __init__(self, model_path: str, device: str = "cuda"):
        from transformers import Blip2Processor, Blip2Model

        self.device = device
        self.processor = Blip2Processor.from_pretrained(model_path)
        self.model = Blip2Model.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).to(device)
        self.model.eval()
        self.embedding_dim = 768

    def encode_text(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self.processor(text=batch, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model.get_text_features(**inputs)
                outputs = outputs / (outputs.norm(dim=-1, keepdim=True) + 1e-12)
                all_embs.append(outputs.cpu().float().numpy())

        return np.vstack(all_embs).astype(np.float32)

    def encode_image(
        self,
        images: Union[Image.Image, List[Image.Image]],
        batch_size: int = 8,
    ) -> np.ndarray:
        if isinstance(images, Image.Image):
            images = [images]

        all_embs = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.model.get_image_features(**inputs)
                outputs = outputs / (outputs.norm(dim=-1, keepdim=True) + 1e-12)
                all_embs.append(outputs.cpu().float().numpy())

        return np.vstack(all_embs).astype(np.float32)
