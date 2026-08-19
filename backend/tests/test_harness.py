"""Tests for the Groq generation harness. No network calls are made."""

from __future__ import annotations

import json
import time

import httpx
import pytest
from pydantic import ValidationError

from app.exceptions import (
    CircuitOpenError,
    MissingCredentialsError,
    UpstreamError,
    UpstreamTimeoutError,
)
from app.harness import (
    CircuitBreaker,
    CircuitState,
    GenerationRequest,
    GroqHarness,
    build_messages,
    is_ungrounded_reply,
)


def sse(*pieces: str) -> bytes:
    """Encode text pieces as a Groq-style server-sent event stream."""
    frames = [
        f"data: {json.dumps({'choices': [{'delta': {'content': piece}}]})}\n\n"
        for piece in pieces
    ]
    frames.append("data: [DONE]\n\n")
    return "".join(frames).encode()


def harness_returning(content: bytes, status: int = 200) -> GroqHarness:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=content)

    return GroqHarness(
        api_key="test-key",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


class TestRequestValidation:
    def test_rejects_empty_query(self) -> None:
        with pytest.raises(ValidationError):
            GenerationRequest(query="", contexts=["ctx"])

    def test_rejects_overlong_query(self) -> None:
        with pytest.raises(ValidationError):
            GenerationRequest(query="a" * 1001, contexts=["ctx"])

    def test_rejects_unknown_language(self) -> None:
        with pytest.raises(ValidationError):
            GenerationRequest(query="hi", contexts=[], language="fr")

    @pytest.mark.parametrize("language", ["en", "hi", "ta"])
    def test_accepts_supported_languages(self, language: str) -> None:
        assert GenerationRequest(query="q", contexts=[], language=language).language == language

    def test_drops_blank_contexts(self) -> None:
        request = GenerationRequest(query="q", contexts=["  ", "real", ""])
        assert request.contexts == ["real"]


class TestPromptAssembly:
    def test_numbers_context_passages(self) -> None:
        messages = build_messages(GenerationRequest(query="q", contexts=["alpha", "beta"]))
        assert "[1] alpha" in messages[1]["content"]
        assert "[2] beta" in messages[1]["content"]

    def test_caps_context_count(self) -> None:
        request = GenerationRequest(query="q", contexts=[f"c{i}" for i in range(20)])
        assert "[5]" not in build_messages(request)[1]["content"]

    def test_system_prompt_names_the_language(self) -> None:
        messages = build_messages(GenerationRequest(query="q", contexts=["c"], language="ta"))
        assert "Tamil" in messages[0]["content"]

    def test_system_prompt_forbids_outside_knowledge(self) -> None:
        content = build_messages(GenerationRequest(query="q", contexts=["c"]))[0]["content"]
        assert "Context not found" in content
        assert "never invent" in content.lower()


class TestUngroundedDetection:
    """Retrieval and the model judge groundedness separately and can disagree."""

    @pytest.mark.parametrize(
        "reply",
        ["Context not found", "Context not found.", "  context not found.  ", "CONTEXT NOT FOUND"],
    )
    def test_detects_refusal_regardless_of_case_or_punctuation(self, reply: str) -> None:
        assert is_ungrounded_reply(reply)

    @pytest.mark.parametrize(
        "reply",
        [
            "A corporation is a legal entity.",
            "",
            # A real answer that merely mentions the phrase must not be
            # mistaken for a refusal.
            "The context not found in the archive was later recovered.",
        ],
    )
    def test_does_not_flag_real_answers(self, reply: str) -> None:
        assert not is_ungrounded_reply(reply)

    @pytest.mark.asyncio
    async def test_result_reports_model_refusal(self) -> None:
        harness = harness_returning(sse("Context not found."))
        result = await harness.generate(GenerationRequest(query="q", contexts=["c"]))
        assert result.model_refused is True
        await harness.aclose()

    @pytest.mark.asyncio
    async def test_result_not_flagged_for_real_answer(self) -> None:
        harness = harness_returning(sse("A corporation is a legal entity."))
        result = await harness.generate(GenerationRequest(query="q", contexts=["c"]))
        assert result.model_refused is False
        await harness.aclose()


class TestGeneration:
    @pytest.mark.asyncio
    async def test_streams_and_joins_pieces(self) -> None:
        harness = harness_returning(sse("A corporation ", "is a legal ", "entity."))
        result = await harness.generate(GenerationRequest(query="q", contexts=["c"]))
        assert result.text == "A corporation is a legal entity."
        assert result.tokens == 3
        await harness.aclose()

    @pytest.mark.asyncio
    async def test_reports_ttft_and_total(self) -> None:
        harness = harness_returning(sse("hello"))
        result = await harness.generate(GenerationRequest(query="q", contexts=["c"]))
        assert result.ttft_ms >= 0
        assert result.total_ms >= result.ttft_ms
        await harness.aclose()

    @pytest.mark.asyncio
    async def test_skips_malformed_frames(self) -> None:
        payload = b'data: {bad json}\n\n' + sse("ok")
        harness = harness_returning(payload)
        result = await harness.generate(GenerationRequest(query="q", contexts=["c"]))
        assert result.text == "ok"
        await harness.aclose()

    @pytest.mark.asyncio
    async def test_empty_stream_is_an_error(self) -> None:
        harness = harness_returning(b"data: [DONE]\n\n")
        with pytest.raises(UpstreamError):
            await harness.generate(GenerationRequest(query="q", contexts=["c"]))
        await harness.aclose()


class TestFailureHandling:
    @pytest.mark.asyncio
    async def test_missing_api_key_raises(self) -> None:
        harness = GroqHarness(api_key="")
        with pytest.raises(MissingCredentialsError):
            await harness.generate(GenerationRequest(query="q", contexts=["c"]))
        await harness.aclose()

    @pytest.mark.asyncio
    async def test_missing_key_never_fabricates_an_answer(self) -> None:
        """A wrong answer is worse than an honest failure."""
        harness = GroqHarness(api_key="")
        with pytest.raises(MissingCredentialsError) as error:
            await harness.generate(GenerationRequest(query="q", contexts=["c"]))
        assert error.value.code == "MISSING_CREDENTIALS"
        await harness.aclose()

    @pytest.mark.asyncio
    async def test_upstream_error_status(self) -> None:
        harness = harness_returning(b'{"error":"rate limited"}', status=429)
        with pytest.raises(UpstreamError):
            await harness.generate(GenerationRequest(query="q", contexts=["c"]))
        await harness.aclose()

    @pytest.mark.asyncio
    async def test_timeout_maps_to_typed_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow", request=request)

        harness = GroqHarness(
            api_key="k", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )
        with pytest.raises(UpstreamTimeoutError):
            await harness.generate(GenerationRequest(query="q", contexts=["c"]))
        await harness.aclose()


class TestCircuitBreaker:
    def test_starts_closed(self) -> None:
        assert CircuitBreaker().state is CircuitState.CLOSED

    def test_opens_after_threshold_failures(self) -> None:
        breaker = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            breaker.record_failure()
        assert breaker.state is CircuitState.OPEN
        with pytest.raises(CircuitOpenError):
            breaker.check()

    def test_stays_closed_below_threshold(self) -> None:
        breaker = CircuitBreaker(failure_threshold=3)
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state is CircuitState.CLOSED
        breaker.check()

    def test_success_resets_failures(self) -> None:
        breaker = CircuitBreaker(failure_threshold=3)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()
        breaker.record_failure()
        assert breaker.state is CircuitState.CLOSED

    def test_half_opens_after_recovery_window(self) -> None:
        breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=0.01)
        breaker.record_failure()
        assert breaker.state is CircuitState.OPEN
        time.sleep(0.02)
        assert breaker.state is CircuitState.HALF_OPEN
        breaker.check()  # probing is allowed

    @pytest.mark.asyncio
    async def test_open_circuit_fails_fast_without_calling_upstream(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(500, content=b"boom")

        harness = GroqHarness(
            api_key="k", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )
        request = GenerationRequest(query="q", contexts=["c"])
        for _ in range(3):
            with pytest.raises(UpstreamError):
                await harness.generate(request)

        calls_before = calls
        with pytest.raises(CircuitOpenError):
            await harness.generate(request)
        assert calls == calls_before, "open circuit must not reach the upstream"
        await harness.aclose()


class TestBudgetReporting:
    @pytest.mark.asyncio
    async def test_within_budget_is_reported_not_enforced(self) -> None:
        """A late answer is still returned; the budget flag tells the truth."""
        harness = harness_returning(sse("answer"))
        result = await harness.generate(GenerationRequest(query="q", contexts=["c"]))
        assert result.text == "answer"
        assert isinstance(result.within_budget, bool)
        await harness.aclose()
