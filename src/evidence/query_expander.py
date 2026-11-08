"""
P-B0: Query Understanding & Expansion (Section I.2, Figure 6).

Takes a detailed image caption and produces:
- expanded_query: enriched description with background context
- core_intent: single most discriminative identity statement
- confidence: self-assessed reliability estimate (used in distillation loss, Eq. 10)
"""

from __future__ import annotations

import json
import re
from typing import Dict, Optional

from src.models.teacher_model import TeacherModel
from src.utils.privacy import redact_external_text


# Corrected prompt template: no path or dataset metadata is provided.
PROMPT_P_B0 = """You are a query understanding and expansion expert. Given a detailed image caption/description, you need to: (1) Understand the core visual characteristics and product identity; (2) Provide grounded semantic expansion; (3) Preserve the original intent; (4) Output in a structured JSON format.

Input Caption: {query}

Please analyze and output in the following JSON format:
{{
  "original_query": "<original caption/query>",
  "expanded_query": "<expanded description with background context, related concepts, and image-specific considerations>",
  "core_intent": "<one sentence describing the main visual identity and distinguishing features>",
  "confidence": 0.9
}}

Requirements:
- expanded_query should add relevant visual and semantic context while preserving all original details.
- core_intent should capture the most distinctive visual characteristics.
- confidence should be a number between 0.0 and 1.0.
- Use only the supplied query. Do not infer from or mention file paths,
  filenames, directory names, class labels, or external metadata.
- Output ONLY the JSON object, no additional text before or after.
- Ensure all strings are properly escaped."""


class QueryExpander:
    """
    Implements P-B0: Query Understanding & Expansion.

    Given a detailed caption (from P-CAP or user-provided text query),
    produces an expanded query, core intent, and confidence score.
    """

    def __init__(self, teacher_model: TeacherModel):
        self.teacher = teacher_model

    def expand_query(self, query: str) -> Dict[str, str]:
        """
        Expand and analyze a query caption.

        Args:
            query: The detailed caption or text query.

        Returns:
            Dictionary with keys: original_query, expanded_query,
            core_intent, confidence.
        """
        query = redact_external_text(query)
        prompt = PROMPT_P_B0.format(query=query)

        response = self.teacher.generate_text(prompt=prompt)
        parsed = self._parse_response(response, query)
        return parsed

    def _parse_response(self, response: str, original_query: str) -> Dict[str, str]:
        """Parse the JSON response from the teacher model."""
        # Try direct JSON parse
        try:
            data = json.loads(response.strip())
            return self._validate_output(data, original_query)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from response
        match = re.search(r'\{.*\}', response, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                return self._validate_output(data, original_query)
            except json.JSONDecodeError:
                pass

        # Fallback: use original query as-is
        return {
            "original_query": original_query,
            "expanded_query": original_query,
            "core_intent": original_query,
            "confidence": 0.5,
        }

    @staticmethod
    def _validate_output(data: Dict, original_query: str) -> Dict[str, str]:
        """Ensure all required fields are present."""
        return {
            "original_query": str(data.get("original_query", original_query)),
            "expanded_query": str(data.get("expanded_query", original_query)),
            "core_intent": str(data.get("core_intent", original_query)),
            "confidence": float(data.get("confidence", 0.9)),
        }
