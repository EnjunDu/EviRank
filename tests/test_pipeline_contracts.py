"""Contract tests for the evidence pipeline and privacy boundaries."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.evidence.evidence_frame import SlotType
from src.evidence.evidence_miner import PROMPT_P_B1
from src.evidence.unified_evidence_generator import UnifiedEvidenceGenerator
from src.evaluate_benchmark import EviRankEvaluator
from src.generate_evidence import EvidenceGenerator
from src.models.teacher_model import TeacherModel
from src.reranking.listwise_ranker import ListwiseRanker
from src.utils.config import load_config


def _evidence_response() -> str:
    slots = {
        slot.value: {"required": [], "forbidden": [], "ignore": []}
        for slot in SlotType
    }
    return json.dumps(
        {
            "generated_caption": "A grounded query description.",
            "original_query": "query",
            "expanded_query": "A grounded query description.",
            "core_intent": "Retrieve the described target.",
            "confidence": 0.9,
            "global_summary": "A grounded query description.",
            **slots,
        }
    )


class CountingTeacher(TeacherModel):
    def __init__(self):
        self.calls = 0
        self.prompts = []

    def _respond(self, prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        if '"ranked_indices"' in prompt:
            return json.dumps(
                {
                    "ranked_indices": [0, 1, 2, 3, 4],
                    "hard_pairs": [],
                    "brief_reasoning": "Ranked using the evidence.",
                }
            )
        return _evidence_response()

    def generate_text(self, prompt: str) -> str:
        return self._respond(prompt)

    def generate_text_from_image(self, image: Image.Image, prompt: str) -> str:
        return self._respond(prompt)

    def generate_text_from_images(self, images, prompt: str) -> str:
        return self._respond(prompt)


class PipelineContractTests(unittest.TestCase):
    def test_unified_evidence_uses_one_call_and_allows_empty_slots(self):
        teacher = CountingTeacher()
        generator = UnifiedEvidenceGenerator(teacher)

        output = generator.generate(
            query_text=(
                "Find /home/private-user/query.jpg and contact "
                "author@example.com with api_key=super-secret-value"
            ),
            query_mode="text-only",
        )

        self.assertEqual(teacher.calls, 1)
        self.assertNotIn("/home/private-user", teacher.prompts[0])
        self.assertNotIn("author@example.com", teacher.prompts[0])
        self.assertNotIn("super-secret-value", teacher.prompts[0])
        for slot in output.evidence_frame.slots.values():
            self.assertEqual(slot.required, [])
            self.assertEqual(slot.forbidden, [])
            self.assertEqual(slot.ignorable, [])

    def test_corrected_b1_prompt_explicitly_allows_empty_lists(self):
        self.assertIn("0-4 sentences", PROMPT_P_B1)
        self.assertIn("output an empty list []", PROMPT_P_B1)
        self.assertNotIn("each list should have 2-4", PROMPT_P_B1)
        self.assertNotIn("image path", PROMPT_P_B1.lower())

    def test_default_online_pipeline_uses_exactly_two_teacher_calls(self):
        config = {
            "models": {"teacher_model": {"type": "disabled"}},
            "reranking": {
                "listwise_top_m": 5,
                "batch_size": 5,
                "allow_divide_and_merge": False,
            },
        }
        evaluator = EviRankEvaluator(config=config, variant="pro", max_workers=1)
        teacher = CountingTeacher()
        evaluator.teacher = teacher
        evaluator.evidence_generator = UnifiedEvidenceGenerator(teacher)
        evaluator.listwise_ranker = ListwiseRanker(
            teacher,
            batch_size=5,
            allow_divide_and_merge=False,
        )

        candidate_ids = [f"raw/private/path/candidate-{i}.jpg" for i in range(20)]
        candidate_captions = [f"Description {i}" for i in range(20)]
        ranked = evaluator.evaluate_query(
            query_caption="A blue floral dress",
            candidate_ids=candidate_ids,
            candidate_captions=candidate_captions,
        )

        self.assertEqual(teacher.calls, 2)
        self.assertEqual(len(ranked), 20)
        self.assertEqual(
            sum(prompt.count("Candidate ") for prompt in teacher.prompts),
            5,
        )
        self.assertTrue(
            all("raw/private/path" not in prompt for prompt in teacher.prompts)
        )

    def test_generated_artifact_does_not_persist_path_or_metadata(self):
        teacher = CountingTeacher()
        generator = EvidenceGenerator.__new__(EvidenceGenerator)
        generator.evidence_generator = UnifiedEvidenceGenerator(teacher)

        with tempfile.TemporaryDirectory() as temp_dir:
            private_dir = Path(temp_dir) / "private-user" / "bicycle_final"
            private_dir.mkdir(parents=True)
            image_path = private_dir / "query.jpg"
            Image.new("RGB", (8, 8), "white").save(image_path)
            result = generator.generate_for_image_query(
                str(image_path),
                metadata={
                    "image_path": str(image_path),
                    "class_id": 123,
                    "email": "author@example.com",
                    "api_key": "do-not-persist",
                },
            )

        serialized = json.dumps(result)
        self.assertEqual(teacher.calls, 1)
        self.assertIn("opaque_id", result)
        self.assertNotIn(str(image_path), serialized)
        self.assertNotIn("private-user", serialized)
        self.assertNotIn("bicycle_final", serialized)
        self.assertNotIn("author@example.com", serialized)
        self.assertNotIn("do-not-persist", serialized)
        self.assertNotIn("class_id", result)

    def test_config_expands_credentials_from_environment(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as file:
            file.write("api_key: ${EVIRANK_TEST_KEY}\n")
            config_path = file.name
        previous = os.environ.get("EVIRANK_TEST_KEY")
        os.environ["EVIRANK_TEST_KEY"] = "runtime-only-value"
        try:
            config = load_config(config_path)
        finally:
            os.unlink(config_path)
            if previous is None:
                os.environ.pop("EVIRANK_TEST_KEY", None)
            else:
                os.environ["EVIRANK_TEST_KEY"] = previous
        self.assertEqual(config["api_key"], "runtime-only-value")


if __name__ == "__main__":
    unittest.main()
