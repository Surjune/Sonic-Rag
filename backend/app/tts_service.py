"""Text to speech via Sarvam bulbul, so an answer can be heard as well as read.

The counterpart to stt_service.py. A user who asked by voice should be able to
receive by voice: making them read the answer to a spoken question is a worse
experience than the one they started with, and in an Indic product it assumes a
literacy that the voice input deliberately did not.

Bulbul rather than the browser's built-in speechSynthesis. Browser voices are
free and instant, but Hindi and Tamil voices are simply absent on most Windows
and many Linux installs -- the machine this was developed on has en-US only --
so the two languages that matter most here would silently fall back to an
English voice reading Devanagari, which is worse than no audio at all.

Synthesis is opt-in per request rather than automatic on every answer. It costs
a network round trip and Sarvam quota, and a user reading text on screen should
not pay for audio they never play.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass

import httpx

from app.config import (
    CONNECT_TIMEOUT_S,
    LANG_TO_SARVAM_CODE,
    SARVAM_TTS_MAX_CHARS,
    SARVAM_TTS_MODEL,
    SARVAM_TTS_SPEAKER,
    SARVAM_TTS_URL,
    STT_TIMEOUT_S,
)
from app.credentials import ROTATABLE_STATUSES, KeyRing
from app.exceptions import MissingCredentialsError, TranscriptionError


@dataclass
class Speech:
    """Synthesized audio, base64 WAV as returned by the API."""

    audio_base64: str
    language: str
    model: str
    speaker: str
    latency_ms: float
    characters: int
    truncated: bool = False


class TextToSpeech:
    """Long-lived client. Create once at startup, reuse per request."""

    def __init__(
        self, api_key: str | None = None, *, client: httpx.AsyncClient | None = None
    ) -> None:
        self._keys = (
            KeyRing.of("SARVAM_API_KEY", api_key)
            if api_key is not None
            else KeyRing.from_env("SARVAM_API_KEY")
        )
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(STT_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
            limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=300.0),
        )

    @property
    def configured(self) -> bool:
        return self._keys.configured

    async def warmup(self) -> bool:
        """Open the TLS connection before the first user asks to hear anything."""
        if not self._keys.configured:
            return False
        try:
            await self._client.get(
                "https://api.sarvam.ai/", timeout=CONNECT_TIMEOUT_S
            )
            return True
        except httpx.HTTPError:
            return False

    async def speak(self, text: str, language: str = "en") -> Speech:
        """Synthesize `text` in `language`. Returns base64 WAV."""
        started = time.perf_counter()

        cleaned = " ".join(text.split())
        if not cleaned:
            raise TranscriptionError(
                "Nothing to speak.", detail="the answer text was empty"
            )

        if not self._keys.configured:
            raise MissingCredentialsError(
                "SARVAM_API_KEY is not configured.",
                detail="Speech output requires the Sarvam key; see .env.example.",
            )

        # Bulbul rejects anything past its limit outright, which would turn a
        # long answer into no audio at all. A truncated reading is more useful
        # than silence, and the caller is told it happened.
        truncated = len(cleaned) > SARVAM_TTS_MAX_CHARS
        if truncated:
            cleaned = cleaned[:SARVAM_TTS_MAX_CHARS]

        payload = {
            "text": cleaned,
            "target_language_code": LANG_TO_SARVAM_CODE.get(language, "en-IN"),
            "speaker": SARVAM_TTS_SPEAKER,
            "model": SARVAM_TTS_MODEL,
        }

        try:
            response = await self._client.post(
                SARVAM_TTS_URL,
                headers={
                    "api-subscription-key": self._keys.active,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        except httpx.HTTPError as error:
            raise TranscriptionError(
                "Speech upstream is unreachable.", detail=str(error)
            ) from error

        if response.status_code >= 400:
            # A throttled or rejected key may have a working backup.
            if response.status_code in ROTATABLE_STATUSES and self._keys.rotate():
                return await self.speak(text, language)
            raise TranscriptionError(
                f"Speech upstream returned {response.status_code}.",
                detail=response.text[:300],
            )

        try:
            body = response.json()
        except ValueError as error:
            raise TranscriptionError("Speech upstream returned a non-JSON body.") from error

        audios = body.get("audios") or []
        if not audios or not str(audios[0]).strip():
            raise TranscriptionError("Speech upstream returned no audio.")

        audio = str(audios[0])
        # Fail here rather than handing the browser a string it cannot decode,
        # where the failure would surface as silent playback with no error.
        try:
            base64.b64decode(audio, validate=True)
        except (ValueError, TypeError) as error:
            raise TranscriptionError(
                "Speech upstream returned malformed audio.", detail=str(error)
            ) from error

        return Speech(
            audio_base64=audio,
            language=language,
            model=SARVAM_TTS_MODEL,
            speaker=SARVAM_TTS_SPEAKER,
            latency_ms=(time.perf_counter() - started) * 1000,
            characters=len(cleaned),
            truncated=truncated,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
