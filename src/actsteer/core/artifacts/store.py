"""Local content-addressed storage for immutable reusable artifacts."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from actsteer.core.artifacts.manifests import (
    ArtifactFile,
    ArtifactManifest,
    canonical_json,
    content_fingerprint,
)


@dataclass(frozen=True)
class ArtifactRef:
    artifact_type: str
    artifact_id: str
    path: Path


def _safe_relative_path(value: str) -> Path:
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise ValueError(f"artifact file path must be safe and relative: {value!r}")
    if candidate.name == "manifest.json":
        raise ValueError("manifest.json is reserved by the artifact store")
    return Path(*candidate.parts)


class ArtifactStore:
    """Write-once artifact directories addressed by their provenance inputs."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def ref(self, artifact_type: str, artifact_id: str) -> ArtifactRef:
        return ArtifactRef(
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            path=self.root / artifact_type / artifact_id,
        )

    def write(
        self,
        artifact_type: str,
        inputs: dict[str, Any],
        files: dict[str, bytes | str | dict[str, Any] | list[Any]],
    ) -> ArtifactRef:
        """Atomically create an artifact or verify an identical existing one."""

        if not artifact_type or "/" in artifact_type or artifact_type in {".", ".."}:
            raise ValueError(f"invalid artifact type: {artifact_type!r}")
        if not files:
            raise ValueError("an artifact must contain at least one payload file")
        prepared: dict[Path, bytes] = {}
        requested_records: list[ArtifactFile] = []
        for relative, payload in sorted(files.items()):
            safe_path = _safe_relative_path(relative)
            if isinstance(payload, bytes):
                content = payload
            elif isinstance(payload, str):
                content = payload.encode("utf-8")
            else:
                content = canonical_json(payload) + b"\n"
            prepared[safe_path] = content
            requested_records.append(
                ArtifactFile(
                    path=safe_path.as_posix(),
                    size=len(content),
                    sha256=hashlib.sha256(content).hexdigest(),
                )
            )
        artifact_id = content_fingerprint(artifact_type, inputs)
        ref = self.ref(artifact_type, artifact_id)
        if ref.path.exists():
            manifest = self.verify(ref)
            if manifest.files != tuple(requested_records):
                raise ValueError(
                    "artifact provenance collides with different payload content; "
                    "include the changed method or code identity in artifact inputs"
                )
            return ref

        parent = ref.path.parent
        parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{artifact_id[:12]}-", dir=parent))
        try:
            for safe_path, content in prepared.items():
                destination = temporary / safe_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
            manifest = ArtifactManifest.create(
                artifact_type,
                inputs,
                tuple(requested_records),
            )
            (temporary / "manifest.json").write_text(
                manifest.model_dump_json(indent=2),
                encoding="utf-8",
            )
            try:
                os.rename(temporary, ref.path)
            except FileExistsError:
                self.verify(ref)
            return ref
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def load_manifest(self, ref: ArtifactRef) -> ArtifactManifest:
        path = ref.path / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"artifact manifest is missing: {path}")
        manifest = ArtifactManifest.model_validate_json(path.read_text(encoding="utf-8"))
        if manifest.artifact_type != ref.artifact_type or manifest.artifact_id != ref.artifact_id:
            raise ValueError(f"artifact identity does not match its path: {ref.path}")
        return manifest

    def verify(self, ref: ArtifactRef) -> ArtifactManifest:
        manifest = self.load_manifest(ref)
        expected_id = content_fingerprint(manifest.artifact_type, manifest.inputs)
        if expected_id != manifest.artifact_id:
            raise ValueError(f"artifact provenance fingerprint is invalid: {ref.path}")
        for record in manifest.files:
            path = ref.path / _safe_relative_path(record.path)
            if not path.is_file():
                raise FileNotFoundError(f"artifact payload is missing: {path}")
            content = path.read_bytes()
            if len(content) != record.size or hashlib.sha256(content).hexdigest() != record.sha256:
                raise ValueError(f"artifact payload checksum failed: {path}")
        return manifest
