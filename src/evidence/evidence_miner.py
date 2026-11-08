"""
P-B1: Structured Evidence Slot Mining (Section I.3, Figure 7).

Decomposes an enriched query into six orthogonal semantic dimensions, each
organized into required/forbidden/ignore constraint lists.

Key design choices (from paper):
(i)  Affirmative-only wording: all constraints are phrased as positive
     statements, including forbidden ones.
(ii) Each list contains 0-4 grounded affirmative sentences and may be empty.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Optional

from src.evidence.evidence_frame import (
    ConstraintLabel,
    EvidenceFrame,
    SlotType,
)
from src.models.teacher_model import TeacherModel
from src.utils.privacy import redact_external_text


# Corrected prompt template: slots and constraint lists may be empty.
PROMPT_P_B1 = """You are a visual evidence extraction expert for image-to-image retrieval. Given a detailed image caption, extract structured evidence that would help identify matching images.

Input Caption: {query}

Your task is to extract evidence into 6 semantic slots. Organize each slot into Required / Forbidden / Ignore lists:

- Required: Affirmative sentences describing visual elements that MUST be present for a correct match.
  Example: "The bicycle is blue with silver handlebars."
- Forbidden: Affirmative sentences describing visual elements that would indicate a WRONG match. These describe what incorrect matches would look like.
  Example: "The bicycle is red with black handlebars."
- Ignore: Affirmative sentences describing similar but non-critical visual elements that can be tolerated.

Scoring logic: score = sum(required_match_scores) - sum(forbidden_match_scores)

Output in the following JSON format:
{{
  "global_summary": "<2-3 sentences providing an overall visual summary>",
  "entities": {{"required":[...], "forbidden":[...], "ignore":[...]}},
  "attributes": {{"required":[...], "forbidden":[...], "ignore":[...]}},
  "actions": {{"required":[...], "forbidden":[...], "ignore":[...]}},
  "relations": {{"required":[...], "forbidden":[...], "ignore":[...]}},
  "scene": {{"required":[...], "forbidden":[...], "ignore":[...]}},
  "key_details": {{"required":[...], "forbidden":[...], "ignore":[...]}}
}}

Requirements:
- All sentences must be affirmative (positive statements), including forbidden items.
- Forbidden sentences describe what wrong matches look like (positive form), NOT using "not" or "should not".
- Each entry should be a complete sentence; each list may have 0-4 sentences.
- If the caption provides no direct evidence for a slot or constraint type,
  output an empty list []. Never invent content to fill a slot.
- Base the evidence strictly on the visual content described in the caption; do not rely on filenames, directories, or any external metadata.
- Sentences should be specific and descriptive for accurate embedding matching.
- Output ONLY valid JSON, no markdown code blocks, no explanations."""


class EvidenceMiner:
    """
    Implements P-B1: Structured Evidence Slot Mining.

    Given a query (output from P-B0), produces an EvidenceFrame with
    six semantic slots and three constraint lists per slot.
    """

    def __init__(self, teacher_model: TeacherModel):
        self.teacher = teacher_model

    def mine_evidence(
        self,
        query: str,
        expanded_query: str = "",
        core_intent: str = "",
        confidence: float = 0.9,
    ) -> EvidenceFrame:
        """
        Mine structured evidence from a query description.

        Args:
            query: Original or expanded query text.
            expanded_query: The expanded query from P-B0.
            core_intent: The core intent from P-B0.
            confidence: Confidence score from P-B0.

        Returns:
            An EvidenceFrame containing all six slots; any slot or list may
            remain empty when unsupported by the query.
        """
        safe_query = redact_external_text(query)
        safe_expanded = redact_external_text(expanded_query)
        safe_core_intent = redact_external_text(core_intent)
        # Use expanded query if available, otherwise original.
        input_text = safe_expanded if safe_expanded else safe_query
        prompt = PROMPT_P_B1.format(query=input_text)

        response = self.teacher.generate_text(prompt=prompt)
        frame = self._parse_response(
            response,
            safe_query,
            safe_expanded,
            safe_core_intent,
            confidence,
        )
        return frame

    def _parse_response(
        self,
        response: str,
        original_query: str,
        expanded_query: str,
        core_intent: str,
        confidence: float,
    ) -> EvidenceFrame:
        """Parse the JSON response into an EvidenceFrame."""
        data = self._extract_json(response)

        if data is None:
            # Fallback: create minimal evidence frame
            frame = EvidenceFrame(
                global_summary=original_query,
                core_intent=core_intent or original_query,
                expanded_query=expanded_query or original_query,
                confidence=confidence,
            )
            return frame

        # Populate EvidenceFrame
        frame = EvidenceFrame(
            global_summary=str(data.get("global_summary", original_query)),
            core_intent=core_intent or original_query,
            expanded_query=expanded_query or original_query,
            confidence=confidence,
        )

        # Parse each slot
        for st in SlotType:
            slot_data = data.get(st.value, {})
            if isinstance(slot_data, dict):
                for text in slot_data.get("required", []):
                    if isinstance(text, str) and text.strip():
                        frame.slots[st].add(text.strip(), ConstraintLabel.REQUIRED)
                for text in slot_data.get("forbidden", []):
                    if isinstance(text, str) and text.strip():
                        frame.slots[st].add(text.strip(), ConstraintLabel.FORBIDDEN)
                for text in slot_data.get("ignore", []):
                    if isinstance(text, str) and text.strip():
                        frame.slots[st].add(text.strip(), ConstraintLabel.IGNORABLE)

        return frame

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        """Extract the first JSON object from text."""
        if not text:
            return None

        # Try direct parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try extracting JSON block
        match = re.search(r'\{.*\}', text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None
