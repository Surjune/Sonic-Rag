"""Sarvam speech-to-text: Indic and English voice input.

Two Sarvam models are used together, issued concurrently:

    saaras:v3     audio -> ENGLISH text + detected source language
    saarika:v2.5  audio -> NATIVE script text

The English text feeds retrieval directly, because the FAISS index is an English
vector space (see indexer.py). The native transcript exists so the interface can
show users their own words in their own script rather than a translation.

They run under asyncio.gather, so the pair costs the slower of the two rather
than their sum: ~500ms instead of ~900ms. The native transcript is optional, and
turning it off halves the request count against a free-tier quota.

Audio arrives as WAV. Browser MediaRecorder defaults to webm/opus, so the
frontend encodes PCM WAV client-side instead; that avoids a server-side ffmpeg
dependency, which would add both latency and a binary that free hosting tiers
do not reliably provide.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import (
    CONNECT_TIMEOUT_S,
    MAX_AUDIO_BYTES,
    SARVAM_API_KEY,
    SARVAM_LANG_MAP,
    SARVAM_TRANSCRIBE_MODEL,
    SARVAM_TRANSCRIBE_URL,
    SARVAM_TRANSLATE_MODEL,
    SARVAM_TRANSLATE_URL,
    STT_TIMEOUT_S,
)
from app.exceptions import InvalidAudioError, MissingCredentialsError, TranscriptionError


@dataclass
class Transcription:
    """Result of one voice input."""

    english_text: str  # what retrieval searches with
    native_text: str  # what the user sees, in their own script
    language: str  # normalized: hi | ta | en
    detected_language_code: str  # raw upstream value, e.g. "hi-IN"
    latency_ms: float


def normalize_language(raw: str | None) -> str:
    """Map Sarvam's locale codes onto the languages the pipeline supports.

    An unsupported language answers in English rather than guessing: the index
    and the answer prompt only cover three languages, and pretending otherwise
    would produce a confidently wrong-language reply.
    """
    if not raw:
        return "en"
    if raw in SARVAM_LANG_MAP:
        return SARVAM_LANG_MAP[raw]
    return SARVAM_LANG_MAP.get(f"{raw.split('-')[0]}-IN", "en")


class SarvamSttService:
    """Long-lived client. Create once at startup, reuse per request."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else SARVAM_API_KEY
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(STT_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
            limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=300.0),
        )

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            raise MissingCredentialsError(
                "SARVAM_API_KEY is not configured.",
                detail="Set SARVAM_API_KEY in the environment; see .env.example.",
            )
        return {"api-subscription-key": self._api_key}

    @staticmethod
    def _validate(audio: bytes) -> None:
        if not audio:
            raise InvalidAudioError("Audio payload is empty.")
        if len(audio) > MAX_AUDIO_BYTES:
            raise InvalidAudioError(
                "Audio payload is too large.",
                detail=f"{len(audio)} bytes exceeds the {MAX_AUDIO_BYTES} byte limit",
            )

    async def _post(self, url: str, audio: bytes, model: str, filename: str) -> dict[str, Any]:
        try:
            response = await self._client.post(
                url,
                headers=self._headers(),
                files={"file": (filename, audio, "audio/wav")},
                data={"model": model},
            )
        except httpx.TimeoutException as error:
            raise TranscriptionError(
                "Speech-to-text upstream timed out.", detail=f"exceeded {STT_TIMEOUT_S}s"
            ) from error
        except httpx.HTTPError as error:
            raise TranscriptionError(
                "Speech-to-text upstream is unreachable.", detail=str(error)
            ) from error

        if response.status_code >= 400:
            raise TranscriptionError(
                f"Speech-to-text upstream returned {response.status_code}.",
                detail=response.text[:300],
            )
        try:
            return dict(response.json())
        except ValueError as error:
            raise TranscriptionError(
                "Speech-to-text upstream returned a non-JSON body."
            ) from error

    async def transcribe(
        self, audio: bytes, *, filename: str = "audio.wav", include_native: bool = True
    ) -> Transcription:
        """Transcribe one utterance.

        The English translation is required; the native transcript is a
        best-effort extra. If only the native call fails the request still
        succeeds, because retrieval needs the English text and the native text
        is only ever displayed.
        """
        self._validate(audio)
        started = time.perf_counter()

        translate = self._post(SARVAM_TRANSLATE_URL, audio, SARVAM_TRANSLATE_MODEL, filename)
        if include_native:
            native = self._post(SARVAM_TRANSCRIBE_URL, audio, SARVAM_TRANSCRIBE_MODEL, filename)
            translated, native_result = await asyncio.gather(
                translate, native, return_exceptions=True
            )
        else:
            translated, native_result = await translate, None

        if isinstance(translated, BaseException):
            raise translated

        english_text = str(translated.get("transcript") or "").strip()
        if not english_text:
            raise TranscriptionError(
                "No speech was recognized in the audio.",
                detail="upstream returned an empty transcript",
            )

        language_code = str(translated.get("language_code") or "")
        native_text = ""
        if isinstance(native_result, dict):
            native_text = str(native_result.get("transcript") or "").strip()
            language_code = language_code or str(native_result.get("language_code") or "")

        return Transcription(
            english_text=english_text,
            native_text=native_text or english_text,
            language=normalize_language(language_code),
            detected_language_code=language_code,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
