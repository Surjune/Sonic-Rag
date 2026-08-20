"""Script detection and text translation for typed Indic queries.

Voice input already arrives in English via saaras (see stt_service.py). Typed
input does not, so a question typed in Devanagari or Tamil has to be translated
before it can be searched against the English vector space.

Script detection runs first and costs microseconds. A Latin-script query skips
the network hop entirely, so English typing -- the common case -- pays nothing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from app.config import (
    CONNECT_TIMEOUT_S,
    LANG_TO_SARVAM_CODE,
    SARVAM_API_KEY,
    SARVAM_TEXT_TRANSLATE_MODEL,
    SARVAM_TEXT_TRANSLATE_URL,
    STT_TIMEOUT_S,
)
from app.credentials import ROTATABLE_STATUSES, KeyRing
from app.exceptions import MissingCredentialsError, TranscriptionError

# Unicode blocks for the scripts the pipeline supports.
DEVANAGARI_RANGE = (0x0900, 0x097F)
TAMIL_RANGE = (0x0B80, 0x0BFF)

# Fraction of letters that must belong to a script before it is claimed. A
# stray Indic character in an otherwise English sentence should not trigger a
# translation round trip.
SCRIPT_SHARE_THRESHOLD = 0.20


def detect_script(text: str) -> str:
    """Return the query language implied by its script: en, hi or ta."""
    letters = 0
    devanagari = 0
    tamil = 0

    for char in text:
        if not char.isalpha():
            continue
        letters += 1
        code = ord(char)
        if DEVANAGARI_RANGE[0] <= code <= DEVANAGARI_RANGE[1]:
            devanagari += 1
        elif TAMIL_RANGE[0] <= code <= TAMIL_RANGE[1]:
            tamil += 1

    if not letters:
        return "en"
    if devanagari / letters >= SCRIPT_SHARE_THRESHOLD:
        return "hi"
    if tamil / letters >= SCRIPT_SHARE_THRESHOLD:
        return "ta"
    return "en"


@dataclass
class Translation:
    text: str
    source_language: str
    latency_ms: float
    translated: bool  # False when the text was already English


class SarvamTranslator:
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

    async def to_english(self, text: str, source_language: str | None = None) -> Translation:
        """Translate to English, skipping the call when already English."""
        started = time.perf_counter()
        language = source_language or detect_script(text)

        if language == "en":
            return Translation(
                text=text,
                source_language="en",
                latency_ms=(time.perf_counter() - started) * 1000,
                translated=False,
            )

        if not self._keys.configured:
            raise MissingCredentialsError(
                "SARVAM_API_KEY is not configured.",
                detail="Translating a typed Indic query requires the Sarvam key.",
            )

        payload = {
            "input": text,
            "source_language_code": LANG_TO_SARVAM_CODE.get(language, "hi-IN"),
            "target_language_code": "en-IN",
            "model": SARVAM_TEXT_TRANSLATE_MODEL,
        }
        try:
            response = await self._client.post(
                SARVAM_TEXT_TRANSLATE_URL,
                headers={"api-subscription-key": self._keys.active, "Content-Type": "application/json"},
                json=payload,
            )
        except httpx.HTTPError as error:
            raise TranscriptionError(
                "Translation upstream is unreachable.", detail=str(error)
            ) from error

        if response.status_code >= 400:
            # A throttled or rejected key may have a working backup.
            if response.status_code in ROTATABLE_STATUSES and self._keys.rotate():
                return await self.to_english(text, source_language)
            raise TranscriptionError(
                f"Translation upstream returned {response.status_code}.",
                detail=response.text[:300],
            )

        try:
            translated_text = str(response.json().get("translated_text") or "").strip()
        except ValueError as error:
            raise TranscriptionError("Translation upstream returned a non-JSON body.") from error

        if not translated_text:
            raise TranscriptionError("Translation upstream returned an empty result.")

        return Translation(
            text=translated_text,
            source_language=language,
            latency_ms=(time.perf_counter() - started) * 1000,
            translated=True,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
