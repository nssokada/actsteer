"""Semantic activation sites, layer selectors, and token selectors."""

from actsteer.core.capture.capture import capture_reduced_forward
from actsteer.core.capture.layers import AbsoluteLayers, RelativeLayers, RelativeWindow
from actsteer.core.capture.reducers import hidden_from_output, masked_mean, replace_hidden
from actsteer.core.capture.sites import ActivationPoint, ActivationSite, ActivationStream
from actsteer.core.capture.tokens import (
    AfterOffset,
    AllValidTokens,
    ExplicitTokenSpan,
    LastKTokens,
    LastValidToken,
    TaggedSpan,
    resolve_token_mask,
)

__all__ = [
    "capture_reduced_forward",
    "AbsoluteLayers",
    "ActivationPoint",
    "ActivationSite",
    "ActivationStream",
    "AfterOffset",
    "AllValidTokens",
    "ExplicitTokenSpan",
    "hidden_from_output",
    "LastKTokens",
    "LastValidToken",
    "masked_mean",
    "RelativeLayers",
    "RelativeWindow",
    "replace_hidden",
    "resolve_token_mask",
    "TaggedSpan",
]
