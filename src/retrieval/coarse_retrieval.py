"""
FAISS-based Coarse Retrieval (Section 4: Implementation Details).

Retrieves Top-K candidates from the image gallery using embedding
similarity (inner product). This is the standard retrieve-then-rerank
prerequisite, not part of EviRank's contribution.

Supported backbones: EVA-CLIP-18B, CLIP-ViT-B/32, CLIP-ViT-L/14,
BLIP-2, DINOv2 (for I->I).
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image


class FAISSRetriever:
    """
    FAISS-based retriever for coarse-stage top-K candidate retrieval.

    Supports both text-to-image and image-to-image retrieval depending
    on the indexed embedding space.
    """

    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        embedding_model=None,
        top_k: int = 20,
    ):
        """
        Args:
            index_path: Path to the FAISS index file.
            metadata_path: Path to the metadata pickle file.
            embedding_model: Optional embedding model for encoding queries.
            top_k: Default number of candidates to retrieve.
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss is required. Install: pip install faiss-cpu (or faiss-gpu)"
            )

        self.embedding_model = embedding_model
        self.top_k = top_k

        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load metadata
        with open(metadata_path, "rb") as f:
            meta_dict = pickle.load(f)

        self.metadata: List[Dict[str, Any]] = meta_dict["metadata"]
        self.dimension: int = meta_dict.get("dimension", self.index.d)

        print(
            f"[FAISSRetriever] Loaded index with {self.index.ntotal} vectors "
            f"(dim={self.dimension})"
        )

    def search(
        self,
        query: Union[str, np.ndarray, Image.Image],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-K candidates for a query.

        Args:
            query: Text string, image, or pre-computed embedding vector.
            top_k: Number of candidates to return.

        Returns:
            List of candidate dictionaries with id, score, caption, image_path.
        """
        if top_k is None:
            top_k = self.top_k

        # Get query embedding
        query_vec = self._get_query_vector(query)

        # Ensure correct shape [1, D]
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        # FAISS search
        distances, indices = self.index.search(
            query_vec.astype(np.float32), top_k
        )

        # Build results
        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.metadata):
                continue

            meta = self.metadata[idx]
            results.append({
                "id": meta.get("image_id", str(idx)),
                "score": float(dist),
                "rank": rank,
                "image_path": meta.get("image_path", ""),
                "caption": (
                    meta["captions"][0] if meta.get("captions") else ""
                ),
                "all_captions": meta.get("captions", []),
            })

        return results

    def search_batch(
        self,
        queries: List[Union[str, np.ndarray]],
        top_k: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Batch retrieval for multiple queries."""
        if top_k is None:
            top_k = self.top_k

        # Get query embeddings
        if isinstance(queries[0], str):
            if self.embedding_model is None:
                raise ValueError("String queries require an embedding_model.")
            query_vecs = self.embedding_model.encode_text(queries)
        else:
            query_vecs = np.vstack(queries)

        # FAISS batch search
        distances, indices = self.index.search(
            query_vecs.astype(np.float32), top_k
        )

        # Build results for each query
        all_results = []
        for q_idx in range(len(queries)):
            results = []
            for rank, (dist, idx) in enumerate(
                zip(distances[q_idx], indices[q_idx])
            ):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                meta = self.metadata[idx]
                results.append({
                    "id": meta.get("image_id", str(idx)),
                    "score": float(dist),
                    "rank": rank,
                    "image_path": meta.get("image_path", ""),
                    "caption": (
                        meta["captions"][0] if meta.get("captions") else ""
                    ),
                    "all_captions": meta.get("captions", []),
                })
            all_results.append(results)

        return all_results

    def _get_query_vector(self, query: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Convert query to embedding vector."""
        if isinstance(query, np.ndarray):
            return query

        if self.embedding_model is None:
            raise ValueError(
                "An embedding_model is required for non-vector queries."
            )

        if isinstance(query, str):
            return self.embedding_model.encode_text(query)
        elif isinstance(query, Image.Image):
            return self.embedding_model.encode_image(query)
        else:
            raise TypeError(f"Unsupported query type: {type(query)}")


def build_faiss_index(
    embeddings: np.ndarray,
    metadata: List[Dict[str, Any]],
    index_path: str,
    metadata_path: str,
    nprobe: int = 32,
) -> None:
    """
    Build and save a FAISS index from embeddings.

    Args:
        embeddings: [N, D] float32 embedding matrix.
        metadata: List of metadata dicts (one per embedding).
        index_path: Output path for FAISS index.
        metadata_path: Output path for metadata pickle.
        nprobe: Number of probes for IVF index.
    """
    import faiss

    embeddings = np.asarray(embeddings, dtype=np.float32)
    n, dim = embeddings.shape

    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path) or ".", exist_ok=True)

    # Choose index type based on dataset size
    if n < 5000:
        index = faiss.IndexFlatIP(dim)
    else:
        nlist = min(1024, max(1, n // 30))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.nprobe = nprobe

    index.add(embeddings)
    faiss.write_index(index, index_path)

    with open(metadata_path, "wb") as f:
        pickle.dump({
            "metadata": metadata,
            "dimension": dim,
            "num_vectors": n,
        }, f)

    print(f"[build_faiss_index] Saved index ({n} vectors, dim={dim}) -> {index_path}")
