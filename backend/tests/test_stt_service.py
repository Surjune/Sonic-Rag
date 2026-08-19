"""Tests for the Sarvam speech-to-text service. No network calls are made."""

from __future__ import annotations

import json

import httpx
import pytest

from app.config import MAX_AUDIO_BYTES
from app.exceptions import InvalidAudioError, MissingCredentialsError, TranscriptionError
from app.stt_service import SarvamSttService, normalize_language

AUDIO = b"RIFF....WAVEfmt " + b"\x00" * 64


def service_with(handler) -> SarvamSttService:
    return SarvamSttService(
        api_key="test-key",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def routed(translate: dict, native: dict | None = None, status: int = 200):
    """Route the translate and transcribe endpoints to canned responses."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "translate" in request.url.path:
            return httpx.Response(status, json=translate)
        return httpx.Response(status, json=native or {})

    return handler


class TestNormalizeLanguage:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("hi-IN", "hi"), ("ta-IN", "ta"), ("en-IN", "en")],
    )
    def test_maps_supported_locales(self, raw: str, expected: str) -> None:
        assert normalize_language(raw) == expected

    def test_bare_code_is_mapped(self) -> None:
        assert normalize_language("hi") == "hi"

    def test_unsupported_language_falls_back_to_english(self) -> None:
        # Answering in a language the index and prompt do not cover would be
        # confidently wrong; English is the honest default.
        assert normalize_language("bn-IN") == "en"

    def test_missing_code_falls_back_to_english(self) -> None:
        assert normalize_language(None) == "en"
        assert normalize_language("") == "en"


class TestAudioValidation:
    @pytest.mark.asyncio
    async def test_empty_audio_rejected(self) -> None:
        service = service_with(routed({"transcript": "x"}))
        with pytest.raises(InvalidAudioError):
            await service.transcribe(b"")
        await service.aclose()

    @pytest.mark.asyncio
    async def test_oversized_audio_rejected(self) -> None:
        service = service_with(routed({"transcript": "x"}))
        with pytest.raises(InvalidAudioError):
            await service.transcribe(b"\x00" * (MAX_AUDIO_BYTES + 1))
        await service.aclose()

    @pytest.mark.asyncio
    async def test_missing_api_key_rejected(self) -> None:
        service = SarvamSttService(api_key="")
        with pytest.raises(MissingCredentialsError):
            await service.transcribe(AUDIO)
        await service.aclose()


class TestTranscription:
    @pytest.mark.asyncio
    async def test_returns_english_native_and_language(self) -> None:
        service = service_with(
            routed(
                {"transcript": "What is a corporation?", "language_code": "hi-IN"},
                {"transcript": "निगम क्या है?", "language_code": "hi-IN"},
            )
        )
        result = await service.transcribe(AUDIO)
        assert result.english_text == "What is a corporation?"
        assert result.native_text == "निगम क्या है?"
        assert result.language == "hi"
        assert result.detected_language_code == "hi-IN"
        assert result.latency_ms >= 0
        await service.aclose()

    @pytest.mark.asyncio
    async def test_native_transcript_can_be_skipped(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            return httpx.Response(200, json={"transcript": "hello", "language_code": "en-IN"})

        service = service_with(handler)
        result = await service.transcribe(AUDIO, include_native=False)
        assert result.english_text == "hello"
        assert len(calls) == 1, "skipping native must not spend a second API call"
        await service.aclose()

    @pytest.mark.asyncio
    async def test_native_failure_does_not_fail_the_request(self) -> None:
        """Retrieval only needs the English text; native is display-only."""

        def handler(request: httpx.Request) -> httpx.Response:
            if "translate" in request.url.path:
                return httpx.Response(200, json={"transcript": "ok", "language_code": "ta-IN"})
            return httpx.Response(500, text="native model down")

        service = service_with(handler)
        result = await service.transcribe(AUDIO)
        assert result.english_text == "ok"
        assert result.native_text == "ok"  # falls back to the English text
        assert result.language == "ta"
        await service.aclose()

    @pytest.mark.asyncio
    async def test_translate_failure_is_fatal(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if "translate" in request.url.path:
                return httpx.Response(503, text="down")
            return httpx.Response(200, json={"transcript": "native"})

        service = service_with(handler)
        with pytest.raises(TranscriptionError):
            await service.transcribe(AUDIO)
        await service.aclose()

    @pytest.mark.asyncio
    async def test_empty_transcript_is_an_error(self) -> None:
        """Silence must not become an empty query that retrieves noise."""
        service = service_with(routed({"transcript": "   ", "language_code": "hi-IN"}))
        with pytest.raises(TranscriptionError):
            await service.transcribe(AUDIO)
        await service.aclose()

    @pytest.mark.asyncio
    async def test_non_json_body_is_an_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>gateway</html>")

        service = service_with(handler)
        with pytest.raises(TranscriptionError):
            await service.transcribe(AUDIO)
        await service.aclose()

    @pytest.mark.asyncio
    async def test_timeout_maps_to_typed_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        service = service_with(handler)
        with pytest.raises(TranscriptionError):
            await service.transcribe(AUDIO)
        await service.aclose()
