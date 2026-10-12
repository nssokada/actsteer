from __future__ import annotations

import unittest

from actsteer.emotion.datasets import from_codrai_records


class CodraiAdapterTests(unittest.TestCase):
    def test_subset_and_limit_are_deterministic(self) -> None:
        expressions = [
            {"emotion": "fear", "story": "Fear story B", "topic": "b"},
            {"emotion": "happy", "story": "Happy story", "topic": "h"},
            {"emotion": "fear", "story": "Fear story A", "topic": "a"},
        ]
        neutral = [
            {"story": "Neutral B", "topic": "b"},
            {"story": "Neutral A", "topic": "a"},
        ]
        first = from_codrai_records(
            expressions,
            neutral,
            concepts=("fear",),
            max_per_concept=1,
        )
        second = from_codrai_records(
            reversed(expressions),
            reversed(neutral),
            concepts=("fear",),
            max_per_concept=1,
        )
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.summary.concepts, {"fear": 1, "neutral": 1})

    def test_missing_requested_concept_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "absent"):
            from_codrai_records(
                [{"emotion": "fear", "story": "A"}],
                [{"story": "N"}],
                concepts=("sad",),
            )


if __name__ == "__main__":
    unittest.main()
