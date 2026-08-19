"""Typed errors with machine-readable codes.

Every failure the API can return maps to one of these, so the HTTP layer can
produce a consistent error envelope instead of leaking a stack trace or, worse,
a plausible-looking fabricated answer.
"""

from __future__ import annotations


class SonicRagError(Exception):
    """Base class. `code` is stable and safe to branch on in the frontend."""

    code = "INTERNAL_ERROR"
    status = 500

    def __init__(self, message: str, *, detail: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "detail": self.detail}


class MissingCredentialsError(SonicRagError):
    """No API key configured.

    Deliberately fatal rather than falling back to a canned answer: a fabricated
    response is worse than an honest failure, because it is indistinguishable
    from a real one.
    """

    code = "MISSING_CREDENTIALS"
    status = 503


class UpstreamTimeoutError(SonicRagError):
    code = "UPSTREAM_TIMEOUT"
    status = 504


class UpstreamError(SonicRagError):
    """The upstream answered, but with an error status or unparseable body.

    `retryable` distinguishes a connection-level fault, where a second attempt
    on a fresh connection genuinely helps, from an upstream that answered and
    said no. Retrying a 429 or a 500 just spends the latency budget twice to
    receive the same refusal.
    """

    code = "UPSTREAM_ERROR"
    status = 502

    def __init__(self, message: str, *, detail: str = "", retryable: bool = False) -> None:
        super().__init__(message, detail=detail)
        self.retryable = retryable


class CircuitOpenError(SonicRagError):
    """Failing fast during cooldown after repeated upstream failures."""

    code = "CIRCUIT_OPEN"
    status = 503


class InvalidAudioError(SonicRagError):
    """Uploaded audio is empty, oversized, or otherwise unusable."""

    code = "INVALID_AUDIO"
    status = 400


class TranscriptionError(SonicRagError):
    """The speech-to-text upstream failed or returned nothing usable."""

    code = "TRANSCRIPTION_FAILED"
    status = 502


class IndexNotLoadedError(SonicRagError):
    """Query arrived before the FAISS artifacts were available."""

    code = "INDEX_NOT_LOADED"
    status = 503
