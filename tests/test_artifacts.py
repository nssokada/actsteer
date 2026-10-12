from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from actsteer.core.artifacts import ArtifactStore


class ArtifactStoreTests(unittest.TestCase):
    def test_write_is_content_addressed_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            first = store.write(
                "vectors",
                {"model": "fake/model", "dataset": "abc"},
                {"metadata.json": {"layers": [2, 4]}, "notes.txt": "complete\n"},
            )
            second = store.write(
                "vectors",
                {"model": "fake/model", "dataset": "abc"},
                {"metadata.json": {"layers": [2, 4]}, "notes.txt": "complete\n"},
            )
            self.assertEqual(first, second)
            self.assertEqual(store.verify(first).artifact_id, first.artifact_id)

    def test_tampering_fails_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            ref = store.write("captures", {"dataset": "abc"}, {"data.bin": b"original"})
            (ref.path / "data.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                store.verify(ref)

    def test_same_provenance_cannot_hide_changed_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            store.write("vectors", {"method": "v1"}, {"data.bin": b"first"})
            with self.assertRaisesRegex(ValueError, "collides"):
                store.write("vectors", {"method": "v1"}, {"data.bin": b"second"})

    def test_unsafe_payload_path_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            with self.assertRaisesRegex(ValueError, "safe and relative"):
                store.write("vectors", {"x": 1}, {"../escape": b"no"})


if __name__ == "__main__":
    unittest.main()
