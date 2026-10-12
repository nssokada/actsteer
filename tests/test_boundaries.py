from __future__ import annotations

import ast
import unittest
from pathlib import Path


class BoundaryTests(unittest.TestCase):
    def test_core_does_not_import_emotion(self) -> None:
        root = Path(__file__).parents[1] / "src" / "actsteer" / "core"
        offenders: list[str] = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                if any(name.startswith("actsteer.emotion") for name in names):
                    offenders.append(str(path.relative_to(root)))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()

