"""Tests for the Sarvam text-to-speech service. No network calls are made."""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from app.config import SARVAM_TTS_MAX_CHARS
from app.exceptions import MissingCredentialsError, TranscriptionError
from app.tts_service import TextToSpeech

WAV = base64.b64encode(b"RIFF....WAVEfmt " + b"\x00" * 64).decode()


def service_with(handler, *, api_key: str | None = "test-key") -> TextToSpeech:
    return TextToSpeech(
        api_key=api_key,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"request_id": "r1", "audios": [WAV]})


class TestSpeak:
    @pytest.mark.asyncio
    async def test_returns_base64_audio(self) -> None:
        speech = await service_with(ok).speak("A corporation is a legal entity.", "en")
        assert speech.audio_base64 == WAV
        assert speech.language == "en"
        assert speech.characters > 0
        assert not speech.truncated

    @pytest.mark.asyncio
    async def test_language_maps_to_the_sarvam_locale(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return ok(request)

        await service_with(handler).speak("निगम क्या है", "hi")
        assert seen["target_language_code"] == "hi-IN"

    @pytest.mark.asyncio
    async def test_unknown_language_falls_back_to_english(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return ok(request)

        await service_with(handler).speak("hello", "fr")
        assert seen["target_language_code"] == "en-IN"

    @pytest.mark.asyncio
    async def test_long_text_is_truncated_rather_than_rejected(self) -> None:
        """Bulbul rejects overlong input outright, which would mean no audio.

        A truncated reading is more useful than silence, so long as the caller
        is told it happened rather than being handed a quietly cut answer.
        """
        speech = await service_with(ok).speak("a " * (SARVAM_TTS_MAX_CHARS), "en")
        assert speech.truncated
        assert speech.characters <= SARVAM_TTS_MAX_CHARS

    @pytest.mark.asyncio
    async def test_empty_text_is_refused_before_the_network(self) -> None:
        def explode(request: httpx.Request) -> httpx.Response:
            raise AssertionError("should not have called the API")

        with pytest.raises(TranscriptionError):
            await service_with(explode).speak("   ", "en")

    @pytest.mark.asyncio
    async def test_missing_key_is_a_typed_error(self) -> None:
        with pytest.raises(MissingCredentialsError):
            await service_with(ok, api_key="").speak("hello", "en")

    @pytest.mark.asyncio
    async def test_upstream_failure_is_typed(self) -> None:
        def failing(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "boom"})

        with pytest.raises(TranscriptionError):
            await service_with(failing).speak("hello", "en")

    @pytest.mark.asyncio
    async def test_empty_audio_list_is_an_error_not_silence(self) -> None:
        def empty(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"audios": []})

        with pytest.raises(TranscriptionError):
            await service_with(empty).speak("hello", "en")

    @pytest.mark.asyncio
    async def test_malformed_base64_fails_here_not_in_the_browser(self) -> None:
        """Otherwise the failure surfaces as silent playback with no error."""

        def bad(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"audios": ["not!valid!base64!"]})

        with pytest.raises(TranscriptionError):
            await service_with(bad).speak("hello", "en")

    @pytest.mark.asyncio
    async def test_unreachable_upstream_is_typed(self) -> None:
        def unreachable(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route", request=request)

        with pytest.raises(TranscriptionError):
            await service_with(unreachable).speak("hello", "en")
