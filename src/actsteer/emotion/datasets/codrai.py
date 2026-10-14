"""Optional adapter for Codrai emotion-expression story tables."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from actsteer.core.datasets import ConceptDataset


def _stable_id(concept: str, text: str) -> str:
    digest = hashlib.sha256(f"{concept}\0{text}".encode()).hexdigest()[:20]
    return f"codrai-{concept}-{digest}"


def from_codrai_records(
    expression_records: Iterable[Mapping[str, Any]],
    neutral_records: Iterable[Mapping[str, Any]],
    *,
    concepts: Sequence[str] | None = None,
    max_per_concept: int | None = None,
    emotion_column: str = "emotion",
    text_column: str = "story",
    neutral_concept: str = "neutral",
) -> ConceptDataset:
    """Normalize Codrai rows without making the core depend on Codrai."""

    if max_per_concept is not None and max_per_concept <= 0:
        raise ValueError("max_per_concept must be positive")
    selected = set(concepts) if concepts is not None else None
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def add(row: Mapping[str, Any], concept: str) -> None:
        if text_column not in row:
            raise ValueError(f"Codrai row is missing text column {text_column!r}")
        text = str(row[text_column]).strip()
        if not text:
            raise ValueError("Codrai story text cannot be blank")
        metadata = {
            key: value
            for key, value in row.items()
            if key not in {emotion_column, text_column}
        }
        grouped[concept].append(
            {
                "id": _stable_id(concept, text),
                "concept": concept,
                "text": text,
                "reference": neutral_concept if concept == neutral_concept else None,
                "metadata": {"source": "codrai", **metadata},
            }
        )

    for row in expression_records:
        if emotion_column not in row:
            raise ValueError(f"Codrai expression row is missing {emotion_column!r}")
        concept = str(row[emotion_column]).strip()
        if selected is None or concept in selected:
            add(row, concept)
    for row in neutral_records:
        add(row, neutral_concept)

    if selected is not None:
        missing = selected - set(grouped)
        if missing:
            raise ValueError(f"requested Codrai concepts are absent: {sorted(missing)}")
    normalized: list[dict[str, Any]] = []
    for concept in sorted(grouped):
        # Hash sorting makes a bounded subset stable if the source row order changes.
        rows = sorted(grouped[concept], key=lambda record: record["id"])
        normalized.extend(rows[:max_per_concept])
    return ConceptDataset.from_records(normalized)


def load_codrai_parquet(
    expression_path: str | Path,
    neutral_path: str | Path,
    **options: Any,
) -> ConceptDataset:
    """Load local Codrai Parquet files through the optional data dependencies."""

    try:
        import pandas as pd
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("Codrai Parquet ingestion requires ActSteer data dependencies") from error
    expression = pd.read_parquet(expression_path).to_dict(orient="records")
    neutral = pd.read_parquet(neutral_path).to_dict(orient="records")
    return from_codrai_records(expression, neutral, **options)
