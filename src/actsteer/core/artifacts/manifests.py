"""Canonical, checksum-verifiable artifact manifests."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def content_fingerprint(artifact_type: str, inputs: dict[str, Any]) -> str:
    payload = {"schema_version": 1, "artifact_type": artifact_type, "inputs": inputs}
    return hashlib.sha256(canonical_json(payload)).hexdigest()


class ArtifactFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    artifact_type: str = Field(min_length=1)
    artifact_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime
    inputs: dict[str, Any]
    files: tuple[ArtifactFile, ...]

    @classmethod
    def create(
        cls,
        artifact_type: str,
        inputs: dict[str, Any],
        files: tuple[ArtifactFile, ...],
    ) -> ArtifactManifest:
        return cls(
            artifact_type=artifact_type,
            artifact_id=content_fingerprint(artifact_type, inputs),
            created_at=datetime.now(UTC),
            inputs=inputs,
            files=files,
        )
