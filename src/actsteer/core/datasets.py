"""Dataset normalization and deterministic content fingerprinting."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from actsteer.core.records import ConceptExample


@dataclass(frozen=True)
class DatasetSummary:
    """Small validation summary safe to log in run manifests."""

    examples: int
    concepts: dict[str, int]
    references: dict[str, int]
    content_modes: dict[str, int]
    fingerprint: str


class ConceptDataset(Sequence[ConceptExample]):
    """An immutable validated collection of normalized concept examples."""

    def __init__(self, examples: Iterable[ConceptExample]) -> None:
        materialized = tuple(examples)
        if not materialized:
            raise ValueError("concept dataset cannot be empty")
        ids = [example.id for example in materialized]
        duplicates = sorted(
            example_id for example_id, count in Counter(ids).items() if count > 1
        )
        if duplicates:
            raise ValueError(f"duplicate example ids: {duplicates[:10]}")
        self._examples = materialized

    @classmethod
    def from_records(cls, records: Iterable[ConceptExample | dict]) -> "ConceptDataset":
        examples: list[ConceptExample] = []
        for index, record in enumerate(records):
            try:
                example = (
                    record
                    if isinstance(record, ConceptExample)
                    else ConceptExample.model_validate(record)
                )
            except ValidationError as error:
                raise ValueError(f"invalid concept record at position {index}") from error
            examples.append(example)
        return cls(examples)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "ConceptDataset":
        """Load normalized :class:`ConceptExample` mappings from JSON Lines."""

        source = Path(path)
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {source}:{line_number}") from error
            if not isinstance(payload, dict):
                raise ValueError(f"record must be a mapping at {source}:{line_number}")
            records.append(payload)
        return cls.from_records(records)

    @classmethod
    def from_tabular_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        id_column: str,
        concept_column: str,
        text_column: str | None = None,
        messages_column: str | None = None,
        reference_column: str | None = None,
        pair_id_column: str | None = None,
        split_column: str | None = None,
        metadata_columns: Sequence[str] = (),
    ) -> "ConceptDataset":
        """Map arbitrary table-like records into the normalized schema.

        Exactly one of ``text_column`` or ``messages_column`` must be supplied.
        The method accepts dictionaries, pandas records, or Hugging Face dataset
        rows without importing either optional library.
        """

        if (text_column is None) == (messages_column is None):
            raise ValueError("provide exactly one of text_column or messages_column")

        normalized: list[dict[str, Any]] = []
        for row_number, row in enumerate(records):
            try:
                payload: dict[str, Any] = {
                    "id": str(row[id_column]),
                    "concept": str(row[concept_column]),
                    "metadata": {column: row[column] for column in metadata_columns},
                }
                if text_column is not None:
                    payload["text"] = row[text_column]
                if messages_column is not None:
                    payload["messages"] = row[messages_column]
                for target, source in (
                    ("reference", reference_column),
                    ("pair_id", pair_id_column),
                    ("split", split_column),
                ):
                    if source is not None:
                        payload[target] = row[source]
            except KeyError as error:
                raise ValueError(
                    f"missing mapped column {error.args[0]!r} in row {row_number}"
                ) from error
            normalized.append(payload)
        return cls.from_records(normalized)

    @classmethod
    def from_parquet(cls, path: str | Path, **mapping: Any) -> "ConceptDataset":
        """Load a Parquet table through the optional pandas/pyarrow extra."""

        try:
            import pandas as pd
        except ImportError as error:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "Parquet ingestion requires the ActSteer data dependencies"
            ) from error
        frame = pd.read_parquet(path)
        return cls.from_tabular_records(frame.to_dict(orient="records"), **mapping)

    def to_jsonl(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(example.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
            for example in self._examples
        ]
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int | slice) -> ConceptExample | tuple[ConceptExample, ...]:
        return self._examples[index]

    def __iter__(self) -> Iterator[ConceptExample]:
        return iter(self._examples)

    @property
    def concepts(self) -> tuple[str, ...]:
        return tuple(sorted({example.concept for example in self._examples}))

    @property
    def summary(self) -> DatasetSummary:
        concepts = Counter(example.concept for example in self._examples)
        references = Counter(example.reference or "<unset>" for example in self._examples)
        modes = Counter(
            "text" if example.text is not None else "messages"
            for example in self._examples
        )
        return DatasetSummary(
            examples=len(self),
            concepts=dict(sorted(concepts.items())),
            references=dict(sorted(references.items())),
            content_modes=dict(sorted(modes.items())),
            fingerprint=self.fingerprint,
        )

    @property
    def fingerprint(self) -> str:
        """Content digest independent of input record ordering."""

        canonical = [
            example.model_dump(mode="json", exclude_none=False)
            for example in sorted(self._examples, key=lambda item: item.id)
        ]
        encoded = json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
