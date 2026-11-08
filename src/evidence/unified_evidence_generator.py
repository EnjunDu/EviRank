"""Single-call query normalization and structured evidence extraction.

P-CAP, P-B0, and P-B1 are logical prompt components, but the deployed
pipeline executes them in one structured teacher request. No file path,
file name, directory name, class label, or dataset identifier is included
in the request.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from PIL import Image

from src.evidence.evidence_frame import (
    ConstraintLabel,
    EvidenceFrame,
    SlotType,
)
from src.models.teacher_model import TeacherModel
from src.utils.privacy import redact_external_text


PROMPT_UNIFIED_EVIDENCE = """You are a multimodal query-understanding and evidence-extraction expert.

Complete query normalization (P-CAP/P-B0) and structured evidence extraction
(P-B1) in this SINGLE response.

QUERY MODE: {query_mode}
QUERY TEXT: {query_text}
MODIFICATION TEXT: {modification_text}
{image_instruction}

Return exactly one JSON object with this schema:
{{
  "generated_caption": "<3-5 grounded sentences for an attached image; otherwise preserve the query text>",
  "original_query": "<the supplied query text>",
  "expanded_query": "<grounded semantic expansion that preserves the original intent>",
  "core_intent": "<one sentence describing the main retrieval intent>",
  "confidence": 0.9,
  "global_summary": "<2-3 sentences summarizing the target>",
  "entities": {{"required": [], "forbidden": [], "ignore": []}},
  "attributes": {{"required": [], "forbidden": [], "ignore": []}},
  "actions": {{"required": [], "forbidden": [], "ignore": []}},
  "relations": {{"required": [], "forbidden": [], "ignore": []}},
  "scene": {{"required": [], "forbidden": [], "ignore": []}},
  "key_details": {{"required": [], "forbidden": [], "ignore": []}}
}}

Rules:
- Each required/forbidden/ignore list contains 0-4 short, complete,
  affirmative, and directly verifiable sentences.
- If the input provides no evidence for a slot or constraint type, output [].
  Never invent constraints to fill a slot.
- Required statements describe what a correct match contains.
- Forbidden statements affirmatively describe what a wrong match contains;
  do not phrase them using "not" or "should not".
- Ignore statements describe grounded variations that should not affect rank.
- For a composed query, preserve grounded reference-image properties and apply
  only the requested modification.
- Use only the supplied text and attached image. Do not infer from or mention
  file paths, file names, directory names, opaque IDs, class labels, dataset
  names, external metadata, or private information.
- Output valid JSON only, with no markdown or explanatory text."""


@dataclass
class UnifiedEvidenceOutput:
    """Parsed output of the single evidence request."""

    generated_caption: str
    expansion: Dict[str, Any]
    evidence_frame: EvidenceFrame

    def to_dict(self) -> Dict[str, Any]:
        """Return the backward-compatible evidence artifact."""
        return {
            "generated_caption": self.generated_caption,
            "b0_expansion": dict(self.expansion),
            "b1_evidence": self.evidence_frame.to_dict(),
        }


class UnifiedEvidenceGenerator:
    """Generate a complete Evidence Frame with exactly one teacher call."""

    def __init__(self, teacher_model: TeacherModel):
        self.teacher = teacher_model

    def generate(
        self,
        query_text: str = "",
        image: Optional[Image.Image] = None,
        modification_text: str = "",
        query_mode: Optional[str] = None,
    ) -> UnifiedEvidenceOutput:
        """Normalize a query and extract evidence in one structured request."""
        query_text = redact_external_text(query_text).strip()
        modification_text = redact_external_text(modification_text).strip()
        mode = query_mode or self._infer_mode(image, modification_text)
        image_instruction = (
            "An image is attached. Ground all visual statements in that image."
            if image is not None
            else "No image is attached. Ground all statements in the supplied text."
        )
        prompt = PROMPT_UNIFIED_EVIDENCE.format(
            query_mode=mode,
            query_text=query_text,
            modification_text=modification_text,
            image_instruction=image_instruction,
        )

        if image is not None:
            response = self.teacher.generate_text_from_image(image=image, prompt=prompt)
        else:
            response = self.teacher.generate_text(prompt=prompt)

        data = self._extract_json(response) or {}
        return self._parse_output(
            data=data,
            original_query=query_text,
            modification_text=modification_text,
        )

    @staticmethod
    def _infer_mode(image: Optional[Image.Image], modification_text: str) -> str:
        if image is not None and modification_text:
            return "composed"
        if image is not None:
            return "image-only"
        return "text-only"

    def _parse_output(
        self,
        data: Dict[str, Any],
        original_query: str,
        modification_text: str,
    ) -> UnifiedEvidenceOutput:
        fallback_query = self._combined_query(original_query, modification_text)
        generated_caption = self._safe_text(
            data.get("generated_caption"),
            fallback_query,
        )
        expanded_query = self._safe_text(
            data.get("expanded_query"),
            fallback_query,
        )
        core_intent = self._safe_text(data.get("core_intent"), fallback_query)
        confidence = self._safe_confidence(data.get("confidence", 0.9))

        frame = EvidenceFrame(
            global_summary=self._safe_text(
                data.get("global_summary"),
                generated_caption or fallback_query,
            ),
            core_intent=core_intent,
            expanded_query=expanded_query,
            confidence=confidence,
        )
        for slot_type in SlotType:
            slot_data = data.get(slot_type.value, {})
            if not isinstance(slot_data, dict):
                continue
            self._add_constraints(
                frame,
                slot_type,
                slot_data.get("required", []),
                ConstraintLabel.REQUIRED,
            )
            self._add_constraints(
                frame,
                slot_type,
                slot_data.get("forbidden", []),
                ConstraintLabel.FORBIDDEN,
            )
            self._add_constraints(
                frame,
                slot_type,
                slot_data.get("ignore", []),
                ConstraintLabel.IGNORABLE,
            )

        expansion = {
            "original_query": original_query,
            "expanded_query": expanded_query,
            "core_intent": core_intent,
            "confidence": confidence,
        }
        return UnifiedEvidenceOutput(generated_caption, expansion, frame)

    @staticmethod
    def _add_constraints(
        frame: EvidenceFrame,
        slot_type: SlotType,
        values: Any,
        label: ConstraintLabel,
    ) -> None:
        """Add up to four non-empty constraints; empty lists are valid."""
        if not isinstance(values, list):
            return
        for value in values[:4]:
            if isinstance(value, str) and value.strip():
                frame.slots[slot_type].add(value.strip(), label)

    @staticmethod
    def _safe_text(value: Any, fallback: str) -> str:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return fallback

    @staticmethod
    def _safe_confidence(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.9

    @staticmethod
    def _combined_query(query_text: str, modification_text: str) -> str:
        if query_text and modification_text:
            return f"Reference: {query_text}. Modification: {modification_text}"
        return modification_text or query_text

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        try:
            data = json.loads(text.strip())
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
