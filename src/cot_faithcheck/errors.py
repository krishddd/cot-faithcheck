"""Exception hierarchy for cot-faithcheck.

All library-raised errors derive from :class:`CotFaithcheckError`, so callers can
catch everything the package throws with a single ``except``.
"""

from __future__ import annotations


class CotFaithcheckError(Exception):
    """Base class for every error raised by cot-faithcheck."""


class TraceParseError(CotFaithcheckError):
    """A trace file / object could not be parsed into reasoning steps."""


class ClientError(CotFaithcheckError):
    """An LLM client failed to produce a completion (network, auth, decoding)."""


class ClientConfigError(ClientError):
    """A client could not be constructed from the given config / environment."""


class NoReasoningStepsError(TraceParseError):
    """The trace parsed successfully but contained no usable reasoning steps."""
