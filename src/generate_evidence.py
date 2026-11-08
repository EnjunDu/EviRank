"""
Evidence Generation Script.

Runs the fused P-CAP/P-B0/P-B1 evidence request on a dataset to produce
pre-computed evidence JSON files for efficient evaluation.

This mirrors the offline evidence mining used in the paper:
- For every query form, normalization and evidence extraction use one request
- For composed queries: combine reference caption + modification text

Usage:
    python -m src.generate_evidence \
        --config configs/default.yaml \
        --dataset sop \
        --input data/sop/test.json \
        --output data/sop/evidence.json \
        --workers 32
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from src.evidence.unified_evidence_generator import UnifiedEvidenceGenerator
from src.models.teacher_model import TeacherModel
from src.utils.config import load_config
from src.utils.privacy import make_opaque_id, safe_error_name


class EvidenceGenerator:
    """
    Generates structured evidence with one fused teacher request per query.

    Paths are used only for local image loading and are never persisted or
    included in teacher prompts.
    """

    def __init__(self, config: Dict, max_workers: int = 32):
        self.teacher = TeacherModel.from_config(config)
        self.evidence_generator = UnifiedEvidenceGenerator(self.teacher)
        self.max_workers = max_workers

        self._lock = Lock()
        self._results: List[Dict] = []
        self._errors: int = 0

    def generate_for_image_query(self, image_path: str, metadata: Dict = None) -> Dict:
        """
        Generate evidence for an image-only query.

        Pipeline: image -> fused P-CAP/P-B0/P-B1 (one teacher call).
        """
        from PIL import Image

        source_id = self._source_id(metadata, image_path)
        result = {"opaque_id": make_opaque_id(source_id, namespace="query")}

        try:
            img = Image.open(image_path).convert("RGB")
            output = self.evidence_generator.generate(
                image=img,
                query_mode="image-only",
            )
            result.update(output.to_dict())

        except Exception as e:
            result["error"] = safe_error_name(e)

        return result

    def generate_for_text_query(self, query_text: str, metadata: Dict = None) -> Dict:
        """
        Generate evidence for a text-only query.

        Pipeline: text -> fused P-B0/P-B1 (one teacher call).
        """
        source_id = self._source_id(metadata, query_text)
        result = {"opaque_id": make_opaque_id(source_id, namespace="query")}

        try:
            output = self.evidence_generator.generate(
                query_text=query_text,
                query_mode="text-only",
            )
            result.update(output.to_dict())

        except Exception as e:
            result["error"] = safe_error_name(e)

        return result

    def generate_for_composed_query(
        self,
        reference_caption: str,
        modification_text: str,
        metadata: Dict = None,
        reference_image_path: str = "",
    ) -> Dict:
        """
        Generate evidence for a composed query (image + text modification).

        The reference image provides PRESERVED required constraints.
        The modification text provides UPDATED required + new forbidden constraints.
        """
        source_id = self._source_id(
            metadata,
            f"{reference_caption}\0{modification_text}",
        )
        result = {"opaque_id": make_opaque_id(source_id, namespace="query")}

        try:
            reference_image = None
            if reference_image_path:
                from PIL import Image
                reference_image = Image.open(reference_image_path).convert("RGB")
            output = self.evidence_generator.generate(
                query_text=reference_caption,
                image=reference_image,
                modification_text=modification_text,
                query_mode="composed",
            )
            result.update(output.to_dict())

        except Exception as e:
            result["error"] = safe_error_name(e)

        return result

    @staticmethod
    def _source_id(metadata: Optional[Dict], fallback: str) -> str:
        """Select an internal identifier without persisting its raw value."""
        metadata = metadata or {}
        for key in ("query_id", "image_id", "reference_id", "id"):
            value = metadata.get(key)
            if value not in (None, ""):
                return str(value)
        return str(fallback)

    def run_batch(
        self,
        items: List[Dict],
        query_type: str,
        output_path: str,
        batch_save_size: int = 64,
    ) -> List[Dict]:
        """
        Process a batch of items concurrently.

        Args:
            items: List of input dicts (format depends on query_type).
            query_type: "image", "text", or "composed".
            output_path: Path for incremental saves.
            batch_save_size: Save every N items.

        Returns:
            List of evidence dicts.
        """
        save_lock = Lock()

        def process_one(item: Dict) -> Dict:
            start = time.time()

            if query_type == "image":
                result = self.generate_for_image_query(
                    image_path=item.get("image_path", item.get("path", "")),
                    metadata=item,
                )
            elif query_type == "text":
                result = self.generate_for_text_query(
                    query_text=item.get("query_text", item.get("caption", "")),
                    metadata=item,
                )
            elif query_type == "composed":
                result = self.generate_for_composed_query(
                    reference_caption=item.get("reference_caption", ""),
                    modification_text=item.get("modification_text", ""),
                    metadata=item,
                    reference_image_path=item.get(
                        "reference_path",
                        item.get("reference_image_path", ""),
                    ),
                )
            else:
                result = {"error": f"Unknown query_type: {query_type}"}

            result["processing_time"] = time.time() - start

            # Incremental save
            with save_lock:
                self._results.append(result)
                if "error" in result:
                    self._errors += 1
                if len(self._results) % batch_save_size == 0:
                    self._save(output_path)

            return result

        # Run concurrently
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_one, item) for item in items]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Mining evidence"):
                pass

        # Final save
        self._save(output_path)
        print(f"Generated {len(self._results)} evidence items ({self._errors} errors)")
        return self._results

    def _save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._results, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="EviRank Evidence Generation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", required=True, help="Input JSON with query items")
    parser.add_argument("--output", required=True, help="Output evidence JSON path")
    parser.add_argument("--query-type", default="image", choices=["image", "text", "composed"])
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--batch-save", type=int, default=64)

    args = parser.parse_args()

    config = load_config(args.config)

    generator = EvidenceGenerator(config=config, max_workers=args.workers)

    with open(args.input) as f:
        items = json.load(f)

    print(f"Loaded {len(items)} items")
    print(f"Query type: {args.query_type}")
    print(f"Workers: {args.workers}")

    generator.run_batch(
        items=items,
        query_type=args.query_type,
        output_path=args.output,
        batch_save_size=args.batch_save,
    )

    print("Evidence generation complete.")


if __name__ == "__main__":
    main()
