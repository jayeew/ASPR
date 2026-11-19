"""Evidence-derived feature selection protocol implementation."""

from .core import EvidenceProtocol, ProtocolError
from .release_registry import (
    DEFAULT_RESULT_RELEASE,
    current_artifact,
    load_current_release,
)

__all__ = [
    "DEFAULT_RESULT_RELEASE",
    "EvidenceProtocol",
    "ProtocolError",
    "current_artifact",
    "load_current_release",
]
