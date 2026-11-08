"""
MLLM Teacher Model (Section 4: Implementation Details).

Supports both closed-source API-based models (e.g., Gemini) and
open-source VLMs (e.g., Qwen-VL) for:
- Image captioning (P-CAP)
- Query expansion (P-B0)
- Evidence mining (P-B1)
- Listwise re-ranking (P-RANK)

The teacher operates in a training-free manner with temperature T=0.15.
"""

from __future__ import annotations

import base64
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from src.utils.config import reject_unresolved_secret_placeholder


def _image_to_base64(
    image: Image.Image,
    max_edge: int = 2048,
    quality: int = 85,
) -> str:
    """Encode PIL image as base64 JPEG string with controlled resolution."""
    # Resize if needed
    w, h = image.size
    if max(w, h) > max_edge:
        scale = max_edge / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _extract_json(text: str) -> Optional[Any]:
    """Extract the first JSON object/array from text."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r'(\[.*\]|\{.*\})', text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return None


class TeacherModel(ABC):
    """Abstract teacher model interface for EviRank pipeline."""

    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        """Generate text response from a text-only prompt."""
        ...

    @abstractmethod
    def generate_text_from_image(
        self, image: Image.Image, prompt: str
    ) -> str:
        """Generate text response given a single image and prompt."""
        ...

    def generate_text_from_images(
        self, images: List[Image.Image], prompt: str
    ) -> str:
        """Generate text response given multiple images and prompt.

        Default implementation uses only the first image.
        Subclasses should override for proper multi-image support.
        """
        if images:
            return self.generate_text_from_image(images[0], prompt)
        return self.generate_text(prompt)

    def score_image_relevance(
        self, query: str, image: Image.Image, caption: str = ""
    ) -> float:
        """Score how well an image matches a query (0-1)."""
        prompt = (
            "You are a strict relevance judge for image retrieval.\n"
            "Given a query and an image, return a single number in [0,1] "
            "representing how well the image matches the query.\n"
            "Output ONLY the number.\n\n"
            f"Query: {query}\n"
        )
        if caption:
            prompt += f"Caption: {caption}\n"

        response = self.generate_text_from_image(image, prompt)
        m = re.search(r"([01](?:\.\d+)?)", response)
        if m:
            return max(0.0, min(1.0, float(m.group(1))))
        return 0.5

    @staticmethod
    def from_config(config: Dict) -> "TeacherModel":
        """Factory method to create teacher from config."""
        if "models" in config:
            tm_cfg = (config.get("models") or {}).get("teacher_model") or {}
        else:
            tm_cfg = config or {}

        tm_type = (tm_cfg.get("type") or "").lower().strip()

        if tm_type in {"closed_source", "closed", "api"}:
            cs = tm_cfg.get("closed_source", tm_cfg)
            api_key = reject_unresolved_secret_placeholder(
                cs.get("api_key"),
                "teacher API key",
            )
            base_url = reject_unresolved_secret_placeholder(
                cs.get("base_url"),
                "teacher API URL",
            )
            return ClosedSourceTeacher(
                model_name=str(cs.get("model_name") or cs.get("name") or ""),
                api_key=api_key,
                base_url=base_url,
                temperature=float(cs.get("temperature", 0.15)),
                timeout_s=float(cs.get("timeout_s", 60.0)),
                max_retries=int(cs.get("max_retries", 0)),
            )

        if tm_type in {"open_source", "open", "hf"}:
            os_cfg = tm_cfg.get("open_source", tm_cfg)
            return OpenSourceTeacher(
                model_name=str(os_cfg.get("model_name") or os_cfg.get("name") or ""),
                device=str(os_cfg.get("device") or "cuda"),
            )

        return DisabledTeacher()


class DisabledTeacher(TeacherModel):
    """Fallback teacher that returns safe defaults (for offline runs)."""

    def generate_text(self, prompt: str) -> str:
        return ""

    def generate_text_from_image(self, image: Image.Image, prompt: str) -> str:
        return ""


@dataclass
class ClosedSourceTeacher(TeacherModel):
    """
    Calls an OpenAI-compatible REST endpoint (e.g., Gemini API proxy).

    Uses httpx for HTTP calls (no dependency on the openai package).
    """
    model_name: str
    api_key: str
    base_url: str
    temperature: float = 0.15
    timeout_s: float = 60.0
    max_retries: int = 0

    def __post_init__(self):
        self._endpoint = self.base_url.rstrip("/") + "/chat/completions"
        self._disabled = not bool(self.base_url)
        self.logical_call_count = 0

    def _post(self, payload: Dict) -> Optional[Dict]:
        if self._disabled:
            return None
        try:
            import httpx
        except ImportError:
            return None

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for _ in range(max(1, self.max_retries + 1)):
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    r = client.post(self._endpoint, headers=headers, json=payload)
                if r.status_code >= 400:
                    continue
                return r.json()
            except Exception:
                continue
        return None

    def _get_response_text(self, data: Optional[Dict]) -> str:
        if not data:
            return ""
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return ""

    def generate_text(self, prompt: str) -> str:
        self.logical_call_count += 1
        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        return self._get_response_text(self._post(payload))

    def generate_text_from_image(self, image: Image.Image, prompt: str) -> str:
        self.logical_call_count += 1
        img_b64 = _image_to_base64(image)
        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        },
                    ],
                }
            ],
        }
        return self._get_response_text(self._post(payload))

    def generate_text_from_images(
        self, images: List[Image.Image], prompt: str
    ) -> str:
        self.logical_call_count += 1
        content = [{"type": "text", "text": prompt}]
        for img in images:
            img_b64 = _image_to_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            })
        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": content}],
        }
        return self._get_response_text(self._post(payload))


class OpenSourceTeacher(TeacherModel):
    """
    Open-source VLM teacher using HuggingFace Transformers.

    Lazily loads the model on first use.
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from transformers import AutoProcessor, AutoModelForVision2Seq

        self._processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_name, trust_remote_code=True
        ).to(self.device)
        self._model.eval()

    def generate_text(self, prompt: str) -> str:
        self._ensure_loaded()
        inputs = self._processor(text=prompt, return_tensors="pt").to(self.device)
        import torch
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=1024)
        return self._processor.decode(outputs[0], skip_special_tokens=True)

    def generate_text_from_image(self, image: Image.Image, prompt: str) -> str:
        self._ensure_loaded()
        inputs = self._processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self.device)
        import torch
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=1024)
        return self._processor.decode(outputs[0], skip_special_tokens=True)
