"""ActSteer public package.

The stable public surface will be promoted here as implementation work units
land. Domain-specific emotion tools live under :mod:`actsteer.emotion` and may
depend on :mod:`actsteer.core`; the reverse dependency is forbidden.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("actsteer")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0.1.0.dev0"

from actsteer.core.config import ActSteerConfig, ArtifactStoreConfig, ModelConfig
from actsteer.core.datasets import ConceptDataset, DatasetSummary
from actsteer.core.records import ChatMessage, ConceptExample

__all__ = [
    "ActSteerConfig",
    "ArtifactStoreConfig",
    "ChatMessage",
    "ConceptDataset",
    "ConceptExample",
    "DatasetSummary",
    "ModelConfig",
    "__version__",
]
