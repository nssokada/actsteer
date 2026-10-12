from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from actsteer.core.datasets import ConceptDataset
from actsteer.core.prompts import PromptSection, PromptSpec, locate_embedded_prompt
from actsteer.core.records import ChatMessage, ConceptExample


class RecordTests(unittest.TestCase):
    def test_requires_exactly_one_content_form(self) -> None:
        with self.assertRaises(ValueError):
            ConceptExample(id="x", concept="afraid")
        with self.assertRaises(ValueError):
            ConceptExample(
                id="x",
                concept="afraid",
                text="story",
                messages=(ChatMessage(role="user", content="story"),),
            )

    def test_plain_and_chat_examples(self) -> None:
        plain = ConceptExample(id="plain", concept="afraid", text="A story")
        chat = ConceptExample(
            id="chat",
            concept="calm",
            messages=(ChatMessage(role="user", content="A message"),),
        )
        self.assertEqual(plain.text, "A story")
        self.assertEqual(chat.messages[0].role, "user")

    def test_dataset_fingerprint_is_order_independent(self) -> None:
        first = ConceptExample(id="a", concept="afraid", text="A")
        second = ConceptExample(id="b", concept="calm", text="B")
        left = ConceptDataset([first, second])
        right = ConceptDataset([second, first])
        self.assertEqual(left.fingerprint, right.fingerprint)

    def test_duplicate_ids_fail(self) -> None:
        example = ConceptExample(id="same", concept="afraid", text="A")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ConceptDataset([example, example])

    def test_jsonl_round_trip(self) -> None:
        dataset = ConceptDataset.from_records(
            [
                {"id": "a", "concept": "afraid", "text": "A"},
                {
                    "id": "b",
                    "concept": "neutral",
                    "text": "B",
                    "reference": "neutral",
                },
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            dataset.to_jsonl(path)
            loaded = ConceptDataset.from_jsonl(path)
        self.assertEqual(dataset.fingerprint, loaded.fingerprint)
        self.assertEqual(loaded.summary.examples, 2)

    def test_tabular_mapping(self) -> None:
        dataset = ConceptDataset.from_tabular_records(
            [{"key": 4, "label": "afraid", "story": "A", "topic": "travel"}],
            id_column="key",
            concept_column="label",
            text_column="story",
            metadata_columns=("topic",),
        )
        self.assertEqual(dataset[0].id, "4")
        self.assertEqual(dataset[0].metadata, {"topic": "travel"})

    def test_missing_tabular_column_fails_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing mapped column"):
            ConceptDataset.from_tabular_records(
                [{"id": "a", "concept": "afraid"}],
                id_column="id",
                concept_column="concept",
                text_column="story",
            )

    def test_prompt_sections_have_stable_spans(self) -> None:
        prompt = PromptSpec(
            sections=(
                PromptSection("environment", "A predator approaches."),
                PromptSection("question", "What do you do?"),
            )
        ).render_plain()
        start, end = prompt.character_spans["environment"]
        self.assertEqual(prompt.text[start:end], "A predator approaches.")

    def test_ambiguous_chat_wrapping_fails(self) -> None:
        prompt = PromptSpec((PromptSection("environment", "forest"),)).render_plain()
        with self.assertRaisesRegex(ValueError, "more than once"):
            locate_embedded_prompt("forest ... forest", prompt)


if __name__ == "__main__":
    unittest.main()
