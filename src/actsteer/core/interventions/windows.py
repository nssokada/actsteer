"""When a steering intervention is active during autoregressive inference."""

from enum import Enum


class InterventionWindow(str, Enum):
    """Canonical temporal protocols for prompt encoding and response decoding."""

    ENCODE_RELEASE = "encode_release"
    RESPONSE_ONLY = "response_only"
    CONTINUOUS = "continuous"
