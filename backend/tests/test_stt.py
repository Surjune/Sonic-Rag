"""Tests for speech-to-text and the Sarvam -> Groq Whisper failover."""

from __future__ import annotations

import httpx
import pytest

from app.exceptions import InvalidAudioError, MissingCredentialsError, TranscriptionError
from app.stt_service import (
    PROVIDER_GROQ,
    PROVIDER_SARVAM,
    GroqWhisperService,
    SarvamSttService,
    SpeechToText,
    normalize_language,
    validate_audio,
)

AUDIO = b"RIFF" + b"\x00" * 200


def sarvam_client(status: int = 200, body: dict | None = None) -> httpx.AsyncClient:
    payload = body if body is not None else {"transcript": "What is a corporation?",
                                             "language_code": "hi-IN"}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def groq_client() -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if "translations" in str(request.url):
            return httpx.Response(200, json={"text": "What is a corporation?"})
        return httpx.Response(200, json={"text": "निगम क्या है?", "language": "hi"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestValidation:
    def test_empty_audio_rejected(self) -> None:
        with pytest.raises(InvalidAudioError):
            validate_audio(b"")

    def test_oversized_audio_rejected(self) -> None:
        with pytest.raises(InvalidAudioError):
            validate_audio(b"x" * 10_000_000)

    def test_normal_audio_passes(self) -> None:
        validate_audio(AUDIO)


class TestLanguageNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"), [("hi-IN", "hi"), ("ta-IN", "ta"), ("en-IN", "en"), ("hi", "hi")]
    )
    def test_maps_known_codes(self, raw: str, expected: str) -> None:
        assert normalize_language(raw) == expected

    def test_unknown_language_defaults_to_english(self) -> None:
        """Answering in a language the index does not cover would be confidently wrong."""
        assert normalize_language("fr-FR") == "en"

    def test_empty_defaults_to_english(self) -> None:
        assert normalize_language(None) == "en"


class TestSarvamProvider:
    @pytest.mark.asyncio
    async def test_transcribes(self) -> None:
        service = SarvamSttService(api_key="k", client=sarvam_client())
        result = await service.transcribe(AUDIO)
        assert result.english_text == "What is a corporation?"
        assert result.language == "hi"
        assert result.provider == PROVIDER_SARVAM
        await service.aclose()

    @pytest.mark.asyncio
    async def test_missing_key_raises(self) -> None:
        service = SarvamSttService(api_key="", client=sarvam_client())
        with pytest.raises(MissingCredentialsError):
            await service.transcribe(AUDIO)
        await service.aclose()

    @pytest.mark.asyncio
    async def test_error_status_raises(self) -> None:
        service = SarvamSttService(api_key="k", client=sarvam_client(status=503))
        with pytest.raises(TranscriptionError):
            await service.transcribe(AUDIO)
        await service.aclose()

    @pytest.mark.asyncio
    async def test_empty_transcript_raises(self) -> None:
        service = SarvamSttService(api_key="k", client=sarvam_client(body={"transcript": ""}))
        with pytest.raises(TranscriptionError):
            await service.transcribe(AUDIO)
        await service.aclose()


class TestGroqWhisperProvider:
    @pytest.mark.asyncio
    async def test_reads_whisper_response_shape(self) -> None:
        """Whisper returns `text`, not Sarvam's `transcript`."""
        service = GroqWhisperService(api_key="k", client=groq_client())
        result = await service.transcribe(AUDIO)
        assert result.english_text == "What is a corporation?"
        assert result.native_text == "निगम क्या है?"
        assert result.provider == PROVIDER_GROQ
        await service.aclose()

    @pytest.mark.asyncio
    async def test_normalizes_bare_iso_language_code(self) -> None:
        """Whisper reports "hi"; the pipeline expects Sarvam's "hi-IN" form."""
        service = GroqWhisperService(api_key="k", client=groq_client())
        result = await service.transcribe(AUDIO)
        assert result.detected_language_code == "hi-IN"
        assert result.language == "hi"
        await service.aclose()


class TestFailover:
    @pytest.mark.asyncio
    async def test_uses_primary_when_it_works(self) -> None:
        facade = SpeechToText(
            SarvamSttService(api_key="k", client=sarvam_client()),
            GroqWhisperService(api_key="g", client=groq_client()),
        )
        result = await facade.transcribe(AUDIO)
        assert result.provider == PROVIDER_SARVAM
        assert result.fallback_reason == ""
        await facade.aclose()

    @pytest.mark.asyncio
    async def test_falls_back_when_primary_has_no_key(self) -> None:
        facade = SpeechToText(
            SarvamSttService(api_key="", client=sarvam_client()),
            GroqWhisperService(api_key="g", client=groq_client()),
        )
        result = await facade.transcribe(AUDIO)
        assert result.provider == PROVIDER_GROQ
        assert "MISSING_CREDENTIALS" in result.fallback_reason
        await facade.aclose()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [401, 403, 500, 503])
    async def test_falls_back_on_upstream_failure(self, status: int) -> None:
        facade = SpeechToText(
            SarvamSttService(api_key="k", client=sarvam_client(status=status)),
            GroqWhisperService(api_key="g", client=groq_client()),
        )
        result = await facade.transcribe(AUDIO)
        assert result.provider == PROVIDER_GROQ
        assert str(status) in result.fallback_reason
        await facade.aclose()

    @pytest.mark.asyncio
    async def test_falls_back_on_timeout(self) -> None:
        def timeout_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow", request=request)

        facade = SpeechToText(
            SarvamSttService(
                api_key="k", client=httpx.AsyncClient(transport=httpx.MockTransport(timeout_handler))
            ),
            GroqWhisperService(api_key="g", client=groq_client()),
        )
        result = await facade.transcribe(AUDIO)
        assert result.provider == PROVIDER_GROQ
        await facade.aclose()

    @pytest.mark.asyncio
    async def test_invalid_audio_does_not_trigger_fallback(self) -> None:
        """Bad audio fails identically on both; retrying only doubles the wait."""
        calls = 0

        def counting(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, json={"text": "x"})

        facade = SpeechToText(
            SarvamSttService(api_key="k", client=sarvam_client()),
            GroqWhisperService(
                api_key="g", client=httpx.AsyncClient(transport=httpx.MockTransport(counting))
            ),
        )
        with pytest.raises(InvalidAudioError):
            await facade.transcribe(b"")
        assert calls == 0
        await facade.aclose()

    @pytest.mark.asyncio
    async def test_both_unconfigured_reports_honestly(self) -> None:
        """No transcript is invented when neither provider is available."""
        facade = SpeechToText(
            SarvamSttService(api_key="", client=sarvam_client()),
            GroqWhisperService(api_key="", client=groq_client()),
        )
        with pytest.raises(MissingCredentialsError) as error:
            await facade.transcribe(AUDIO)
        assert "No speech-to-text provider" in error.value.message
        await facade.aclose()

    @pytest.mark.asyncio
    async def test_reports_provider_availability(self) -> None:
        facade = SpeechToText(
            SarvamSttService(api_key="", client=sarvam_client()),
            GroqWhisperService(api_key="g", client=groq_client()),
        )
        assert facade.providers == {PROVIDER_SARVAM: False, PROVIDER_GROQ: True}
        assert facade.configured is True
        await facade.aclose()

    @pytest.mark.asyncio
    async def test_fallback_latency_includes_the_failed_attempt(self) -> None:
        """The user waited for the primary too; hiding that would understate it."""
        facade = SpeechToText(
            SarvamSttService(api_key="k", client=sarvam_client(status=500)),
            GroqWhisperService(api_key="g", client=groq_client()),
        )
        result = await facade.transcribe(AUDIO)
        assert result.latency_ms > 0
        await facade.aclose()
