"""Speech-to-text with a two-vendor failover chain.

    Sarvam (primary)   -> Groq Whisper (fallback)

Sarvam is purpose-built for Indic speech and reports the detected language, so
it leads. Whisper on Groq stands by for when Sarvam has no key, rejects the
key, or fails: two providers on different vendors means a single outage does
not take voice input down mid-demo. Which provider answered is recorded and
returned, because a silent fallback that nobody can see is a debugging trap.

Both providers are asked for two things at once:

    English text   feeds retrieval, because the FAISS index is an English
                   vector space (see indexer.py)
    native text    shown to the user in their own script, never a translation

The pair is issued under asyncio.gather, so it costs the slower of the two
rather than their sum. The native transcript is best-effort: if only that call
fails the request still succeeds, since retrieval needs the English text and
the native text is only ever displayed.

Audio arrives as WAV. Browser MediaRecorder defaults to webm/opus, so the
frontend encodes PCM WAV client-side; that avoids a server-side ffmpeg
dependency, which would add latency and a binary free hosting tiers do not
reliably provide.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import (
    CONNECT_TIMEOUT_S,
    GROQ_API_KEY,
    GROQ_TRANSCRIBE_URL,
    GROQ_TRANSLATE_URL,
    GROQ_WHISPER_MODEL,
    GROQ_WHISPER_TRANSLATE_MODEL,
    MAX_AUDIO_BYTES,
    SARVAM_API_KEY,
    SARVAM_LANG_MAP,
    SARVAM_TRANSCRIBE_MODEL,
    SARVAM_TRANSCRIBE_URL,
    SARVAM_TRANSLATE_MODEL,
    SARVAM_TRANSLATE_URL,
    STT_TIMEOUT_S,
)
from app.credentials import ROTATABLE_STATUSES, KeyRing
from app.exceptions import InvalidAudioError, MissingCredentialsError, TranscriptionError

logger = logging.getLogger(__name__)

PROVIDER_SARVAM = "sarvam"
PROVIDER_GROQ = "groq_fallback"


@dataclass
class Transcription:
    """Result of one voice input."""

    english_text: str  # what retrieval searches with
    native_text: str  # what the user sees, in their own script
    language: str  # normalized: hi | ta | en
    detected_language_code: str  # raw upstream value, e.g. "hi-IN"
    latency_ms: float
    provider: str = PROVIDER_SARVAM
    fallback_reason: str = ""  # why the primary was skipped, if it was


def normalize_language(raw: str | None) -> str:
    """Map upstream locale codes onto the languages the pipeline supports.

    An unsupported language answers in English rather than guessing: the index
    and the answer prompt only cover three languages, and pretending otherwise
    would produce a confidently wrong-language reply.
    """
    if not raw:
        return "en"
    if raw in SARVAM_LANG_MAP:
        return SARVAM_LANG_MAP[raw]
    return SARVAM_LANG_MAP.get(f"{raw.split('-')[0]}-IN", "en")


def validate_audio(audio: bytes) -> None:
    if not audio:
        raise InvalidAudioError("Audio payload is empty.")
    if len(audio) > MAX_AUDIO_BYTES:
        raise InvalidAudioError(
            "Audio payload is too large.",
            detail=f"{len(audio)} bytes exceeds the {MAX_AUDIO_BYTES} byte limit",
        )


class _HttpSpeechProvider:
    """Shared multipart POST handling for both vendors."""

    def __init__(self, keys: KeyRing, client: httpx.AsyncClient | None = None) -> None:
        self._keys = keys
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(STT_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
            limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=300.0),
        )

    @property
    def configured(self) -> bool:
        return self._keys.configured

    @property
    def key_label(self) -> str:
        return self._keys.label

    def _headers(self) -> dict[str, str]:
        raise NotImplementedError

    async def _post(
        self, url: str, audio: bytes, filename: str, data: dict[str, str]
    ) -> dict[str, Any]:
        """POST, rotating to a backup key if this one is rejected or throttled.

        Quota on a free tier is per-key, so a second key on the same vendor
        recovers from a 429 that switching vendors cannot.
        """
        while True:
            try:
                return await self._post_once(url, audio, filename, data)
            except TranscriptionError as error:
                if error.status_code not in ROTATABLE_STATUSES or not self._keys.rotate():
                    raise
                logger.warning(
                    "%s rejected with %s, rotating to %s",
                    self._keys.name, error.status_code, self._keys.label,
                )

    async def _post_once(
        self, url: str, audio: bytes, filename: str, data: dict[str, str]
    ) -> dict[str, Any]:
        try:
            response = await self._client.post(
                url,
                headers=self._headers(),
                files={"file": (filename, audio, "audio/wav")},
                data=data,
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
            failure = TranscriptionError(
                f"Speech-to-text upstream returned {response.status_code}.",
                detail=response.text[:300],
            )
            failure.status_code = response.status_code
            raise failure
        try:
            return dict(response.json())
        except ValueError as error:
            raise TranscriptionError("Speech-to-text upstream returned a non-JSON body.") from error

    async def warmup(self, url: str) -> bool:
        """Open a TLS connection before the first real request.

        The first HTTPS call to a host pays DNS resolution plus a full
        handshake. Doing that at startup instead of during someone's first
        spoken question removes a one-off spike of a few hundred milliseconds
        from the number they actually see.

        A HEAD on the endpoint is enough to establish the connection; the
        status is irrelevant, since the pooled socket is the point.
        """
        if not self.configured:
            return False
        try:
            await self._client.head(url, timeout=CONNECT_TIMEOUT_S)
        except httpx.HTTPError:
            # A cold connection is a slower first request, never a failure.
            return False
        return True

    async def aclose(self) -> None:
        await self._client.aclose()


class SarvamSttService(_HttpSpeechProvider):
    """Primary provider. saaras translates to English, saarika keeps the script."""

    name = PROVIDER_SARVAM

    def __init__(
        self, api_key: str | None = None, *, client: httpx.AsyncClient | None = None
    ) -> None:
        keys = (
            KeyRing.of("SARVAM_API_KEY", api_key)
            if api_key is not None
            else KeyRing.from_env("SARVAM_API_KEY")
        )
        super().__init__(keys, client)

    def _headers(self) -> dict[str, str]:
        if not self._keys.configured:
            raise MissingCredentialsError(
                "SARVAM_API_KEY is not configured.",
                detail="Set SARVAM_API_KEY in the environment; see .env.example.",
            )
        return {"api-subscription-key": self._keys.active}

    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "audio.wav",
        include_native: bool = True,
        language_hint: str | None = None,
    ) -> Transcription:
        # The hint is accepted for interface parity and ignored: saaras detects
        # the source language reliably on its own, and forcing it would discard
        # that signal when the caller guesses wrong.
        del language_hint
        validate_audio(audio)
        started = time.perf_counter()

        translate = self._post(
            SARVAM_TRANSLATE_URL, audio, filename, {"model": SARVAM_TRANSLATE_MODEL}
        )
        if include_native:
            native = self._post(
                SARVAM_TRANSCRIBE_URL, audio, filename, {"model": SARVAM_TRANSCRIBE_MODEL}
            )
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
            provider=self.name,
        )


class GroqWhisperService(_HttpSpeechProvider):
    """Fallback provider on Groq's OpenAI-compatible audio endpoints."""

    name = PROVIDER_GROQ

    def __init__(
        self, api_key: str | None = None, *, client: httpx.AsyncClient | None = None
    ) -> None:
        keys = (
            KeyRing.of("GROQ_API_KEY", api_key)
            if api_key is not None
            else KeyRing.from_env("GROQ_API_KEY")
        )
        super().__init__(keys, client)

    def _headers(self) -> dict[str, str]:
        if not self._keys.configured:
            raise MissingCredentialsError(
                "GROQ_API_KEY is not configured.",
                detail="Speech fallback needs GROQ_API_KEY; see .env.example.",
            )
        return {"Authorization": f"Bearer {self._keys.active}"}

    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "audio.wav",
        include_native: bool = True,
        language_hint: str | None = None,
    ) -> Transcription:
        validate_audio(audio)
        started = time.perf_counter()

        # Translation must run on large-v3: the turbo variant is
        # transcription-only and rejects the translations endpoint.
        translate = self._post(
            GROQ_TRANSLATE_URL,
            audio,
            filename,
            {"model": GROQ_WHISPER_TRANSLATE_MODEL, "response_format": "json"},
        )
        if include_native:
            # verbose_json carries the detected language, which plain json omits.
            native_form = {"model": GROQ_WHISPER_MODEL, "response_format": "verbose_json"}
            if language_hint:
                # Whisper hears spoken Hindi as Urdu and returns Arabic script,
                # which is unreadable to a Hindi speaker. When the caller has
                # already chosen a language, that is better evidence than the
                # model's guess.
                native_form["language"] = language_hint[:2]
            native = self._post(GROQ_TRANSCRIBE_URL, audio, filename, native_form)
            translated, native_result = await asyncio.gather(
                translate, native, return_exceptions=True
            )
        else:
            translated, native_result = await translate, None

        if isinstance(translated, BaseException):
            raise translated

        english_text = str(translated.get("text") or "").strip()
        if not english_text:
            raise TranscriptionError(
                "No speech was recognized in the audio.",
                detail="fallback provider returned an empty transcript",
            )

        native_text = ""
        language_code = ""
        if isinstance(native_result, dict):
            native_text = str(native_result.get("text") or "").strip()
            # Whisper reports bare ISO codes ("hi"); normalize_language expects
            # the locale form Sarvam uses.
            detected = str(native_result.get("language") or "").strip().lower()
            if language_hint:
                language_code = f"{language_hint[:2]}-IN"
            elif detected:
                language_code = detected if "-" in detected else f"{detected[:2]}-IN"

        return Transcription(
            english_text=english_text,
            native_text=native_text or english_text,
            language=normalize_language(language_code),
            detected_language_code=language_code,
            latency_ms=(time.perf_counter() - started) * 1000,
            provider=self.name,
        )


class SpeechToText:
    """Failover facade: try Sarvam, fall back to Groq Whisper.

    Only the primary's own failures trigger the fallback. InvalidAudioError is
    re-raised untouched, because empty or oversized audio will fail identically
    on the second provider and retrying it just doubles the wait.
    """

    def __init__(
        self,
        primary: SarvamSttService | None = None,
        fallback: GroqWhisperService | None = None,
    ) -> None:
        self._primary = primary if primary is not None else SarvamSttService()
        self._fallback = fallback if fallback is not None else GroqWhisperService()

    @property
    def configured(self) -> bool:
        return self._primary.configured or self._fallback.configured

    @property
    def providers(self) -> dict[str, bool]:
        return {
            PROVIDER_SARVAM: self._primary.configured,
            PROVIDER_GROQ: self._fallback.configured,
        }

    async def warmup(self) -> dict[str, bool]:
        """Pre-connect both providers concurrently during startup."""
        primary, fallback = await asyncio.gather(
            self._primary.warmup(SARVAM_TRANSLATE_URL),
            self._fallback.warmup(GROQ_TRANSCRIBE_URL),
            return_exceptions=True,
        )
        return {
            PROVIDER_SARVAM: primary is True,
            PROVIDER_GROQ: fallback is True,
        }

    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "audio.wav",
        include_native: bool = True,
        language_hint: str | None = None,
    ) -> Transcription:
        validate_audio(audio)
        started = time.perf_counter()
        reason = ""

        if self._primary.configured:
            try:
                result = await self._primary.transcribe(
                    audio,
                    filename=filename,
                    include_native=include_native,
                    language_hint=language_hint,
                )
                logger.info("stt provider=%s latency_ms=%.0f", result.provider, result.latency_ms)
                return result
            except InvalidAudioError:
                raise
            except (TranscriptionError, MissingCredentialsError) as error:
                reason = f"{error.code}: {error.message}"
                logger.warning("stt primary failed, falling back: %s", reason)
        else:
            reason = "MISSING_CREDENTIALS: SARVAM_API_KEY is not configured"
            logger.warning("stt primary unconfigured, using fallback")

        if not self._fallback.configured:
            # Both unavailable: report honestly rather than inventing a transcript.
            raise MissingCredentialsError(
                "No speech-to-text provider is configured.",
                detail=f"Sarvam unavailable ({reason}) and GROQ_API_KEY is not set.",
            )

        result = await self._fallback.transcribe(
            audio, filename=filename, include_native=include_native, language_hint=language_hint
        )
        elapsed = (time.perf_counter() - started) * 1000
        logger.info("stt provider=%s latency_ms=%.0f reason=%s", result.provider, elapsed, reason)
        return Transcription(
            english_text=result.english_text,
            native_text=result.native_text,
            language=result.language,
            detected_language_code=result.detected_language_code,
            # The primary's failed attempt is part of what the user waited for.
            latency_ms=elapsed,
            provider=result.provider,
            fallback_reason=reason,
        )

    async def aclose(self) -> None:
        await asyncio.gather(
            self._primary.aclose(), self._fallback.aclose(), return_exceptions=True
        )
