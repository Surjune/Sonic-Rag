"""Groq LPU generation harness: validation, streaming, retries, circuit breaker.

Uses httpx directly rather than the Groq SDK. On a latency-critical path the
connection pool must be explicit and long-lived: a reused TLS connection saves
the full handshake on every request, which is worth more than the entire local
compute budget. Per-phase timeouts also need to be set independently, which the
SDK does not expose as directly.

Time-to-first-token is the metric that matters here. A streamed answer is
readable the moment the first token lands, so TTFT is what the user experiences
as "fast" -- total completion time is a throughput number, not a latency one.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Sequence

import httpx
from pydantic import BaseModel, Field, field_validator

from app.config import (
    CIRCUIT_FAILURE_THRESHOLD,
    CIRCUIT_RECOVERY_S,
    CONNECT_TIMEOUT_S,
    GROQ_API_KEY,
    GROQ_API_URL,
    GROQ_MODEL,
    LLM_PROVIDER,
    OLLAMA_API_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    MAX_CONTEXT_CHUNKS,
    MAX_OUTPUT_TOKENS,
    MAX_RETRIES,
    REQUEST_TIMEOUT_S,
    TTFT_BUDGET_MS,
    UNGROUNDED_MESSAGE,
)
from app.credentials import ROTATABLE_STATUSES, KeyRing
from app.exceptions import (
    CircuitOpenError,
    MissingCredentialsError,
    UpstreamError,
    UpstreamTimeoutError,
)
from app.tools import ToolRegistry, ToolResult, parse_tool_calls

# Cap on model -> tool -> model cycles. Without it a model that keeps requesting
# tools loops until the request times out; three rounds is well beyond what a
# grounded lookup needs.
MAX_TOOL_ROUNDS = 3

logger = logging.getLogger(__name__)

LANGUAGE_NAMES: dict[str, str] = {"en": "English", "hi": "Hindi", "ta": "Tamil"}

SYSTEM_PROMPT = (
    "You answer strictly from the numbered context passages provided. "
    "The question may use different wording, spelling or grammar than the passages, "
    "especially when it came from speech recognition or translation. Answer whenever "
    "the passages cover the same subject, even if the exact word does not appear. "
    f"Only if the passages genuinely do not address the subject, reply exactly: "
    f"{UNGROUNDED_MESSAGE}. "
    "Never use outside knowledge and never invent details. "
    # The observed failure was not a wholly invented answer -- it was a real
    # definition lifted from a passage, followed by several paragraphs of
    # advice from the model's own training that no passage contained. Partial
    # padding is the harder case, because the grounded opening makes the rest
    # look sourced, so the prompt has to rule out the continuation explicitly
    # rather than only forbidding invention in general.
    "Do not add advice, steps, recommendations, examples or background that the "
    "passages do not state, even when you know them to be true and even when they "
    "would be helpful. If the passages define a term but do not answer what was "
    f"asked about it, say {UNGROUNDED_MESSAGE} rather than defining the term and "
    "continuing from your own knowledge. "
    "Answer in {language}. Be concise: two or three sentences."
)


def is_ungrounded_reply(text: str) -> bool:
    """Whether the model itself declined for lack of usable context.

    Retrieval and the model are two independent judges of groundedness and they
    can disagree: a passage can clear the cosine threshold while the model still
    finds it unusable. Detecting the refusal lets the caller report what actually
    happened instead of labelling a refusal as a grounded answer.
    """
    normalized = text.strip().rstrip(".").casefold()
    return normalized == UNGROUNDED_MESSAGE.rstrip(".").casefold()


class GenerationRequest(BaseModel):
    """Validated at the boundary, before any network call is made."""

    query: str = Field(min_length=1, max_length=1000)
    contexts: list[str] = Field(default_factory=list)
    language: str = Field(default="en")

    @field_validator("language")
    @classmethod
    def _known_language(cls, value: str) -> str:
        if value not in LANGUAGE_NAMES:
            raise ValueError(f"language must be one of {sorted(LANGUAGE_NAMES)}")
        return value

    @field_validator("contexts")
    @classmethod
    def _drop_blank_contexts(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item and item.strip()]


@dataclass
class GenerationResult:
    """What the harness reports back, including honest timing."""

    text: str
    ttft_ms: float
    total_ms: float
    model: str
    tokens: int = 0
    within_budget: bool = False
    truncated: bool = False
    # True when the model itself declined for lack of usable context, even
    # though retrieval cleared the similarity threshold.
    model_refused: bool = False
    # Tools the model invoked, in order, with their timings.
    tool_results: list[ToolResult] = field(default_factory=list)
    tool_rounds: int = 0


class CircuitState(str, Enum):
    CLOSED = "CLOSED"  # normal
    OPEN = "OPEN"  # failing fast
    HALF_OPEN = "HALF_OPEN"  # probing recovery


@dataclass
class CircuitBreaker:
    """Fail fast after repeated upstream failures.

    Without this, every user waits the full timeout for an upstream that is
    already known to be down, turning one outage into a queue of slow requests.
    """

    failure_threshold: int = CIRCUIT_FAILURE_THRESHOLD
    recovery_seconds: float = CIRCUIT_RECOVERY_S
    failures: int = 0
    opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        if self.opened_at is None:
            return CircuitState.CLOSED
        if time.monotonic() - self.opened_at >= self.recovery_seconds:
            return CircuitState.HALF_OPEN
        return CircuitState.OPEN

    def check(self) -> None:
        if self.state is CircuitState.OPEN:
            assert self.opened_at is not None
            remaining = self.recovery_seconds - (time.monotonic() - self.opened_at)
            raise CircuitOpenError(
                "Generation upstream is unavailable; retrying shortly.",
                detail=f"circuit open, {remaining:.1f}s remaining",
            )

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = time.monotonic()


def build_messages(request: GenerationRequest) -> list[dict[str, str]]:
    """Assemble the prompt. Context is numbered so the model can cite it."""
    language = LANGUAGE_NAMES[request.language]
    chunks = request.contexts[:MAX_CONTEXT_CHUNKS]
    context_block = "\n\n".join(
        f"[{position}] {text}" for position, text in enumerate(chunks, start=1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(language=language)},
        {
            "role": "user",
            "content": f"Context:\n{context_block}\n\nQuestion: {request.query}",
        },
    ]


class GroqHarness:
    """Long-lived client wrapper. Create once at startup, reuse per request."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        model: str | None = None,
        tools: ToolRegistry | None = None,
        provider: str | None = None,
    ) -> None:
        # Groq is the default and the deployment target. Ollama is opt-in and
        # exists to measure what removing the network hop actually buys; it
        # speaks the same OpenAI-compatible protocol, so only the endpoint,
        # the model name and the auth requirement differ.
        self._provider = (provider or LLM_PROVIDER).strip().lower()
        self._local = self._provider == "ollama"
        self._api_url = OLLAMA_API_URL if self._local else GROQ_API_URL

        self._keys = (
            KeyRing.of("GROQ_API_KEY", api_key)
            if api_key is not None
            else KeyRing.from_env("GROQ_API_KEY")
        )
        if model is not None:
            self._model = model
        else:
            self._model = OLLAMA_MODEL if self._local else GROQ_MODEL
        self._tools = tools or ToolRegistry()
        self._breaker = CircuitBreaker()
        # Live re-pin tasks, held so the loop cannot collect them mid-flight.
        self._pending: set[asyncio.Task[bool]] = set()
        # A shared pool with keep-alive: the saved TLS handshake on a warm
        # connection outweighs the entire local compute budget.
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(REQUEST_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
            limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=300.0),
        )

    @property
    def circuit_state(self) -> CircuitState:
        return self._breaker.state

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    @property
    def configured(self) -> bool:
        # A local provider needs no key, so credentials are not what decides
        # whether it is usable.
        return True if self._local else self._keys.configured

    @property
    def key_label(self) -> str:
        return self._keys.label

    def _rotate_on(self, status: int) -> bool:
        """Try a backup key when this one is rejected or throttled."""
        if status not in ROTATABLE_STATUSES or not self._keys.rotate():
            return False
        logger.warning("GROQ_API_KEY rejected with %s, rotating to %s", status, self._keys.label)
        return True

    def _headers(self) -> dict[str, str]:
        # A local server has no credential to present, and demanding one would
        # make the comparison impossible to run.
        if self._local:
            return {"Content-Type": "application/json"}
        if not self._keys.configured:
            raise MissingCredentialsError(
                "GROQ_API_KEY is not configured.",
                detail="Set GROQ_API_KEY in the environment; see .env.example.",
            )
        return {
            "Authorization": f"Bearer {self._keys.active}",
            "Content-Type": "application/json",
        }

    async def pin(self) -> bool:
        """Load the local model into VRAM and keep it there.

        Ollama evicts after five minutes idle, and the reload costs 8155ms
        against 22ms warm -- rare enough to look like a glitch and slow enough
        to ruin a demo. An empty prompt to the native endpoint loads the model
        without generating anything, and only that endpoint honours
        keep_alive: the OpenAI-compatible one used for generation ignores it
        and resets the timer to five minutes on every request.

        Fails quietly. Ollama not running is the ordinary case for anyone who
        has not installed it, and is not a problem worth surfacing.
        """
        if not self._local:
            return False
        try:
            base = self._api_url.split("/v1/")[0]
            response = await self._client.post(
                f"{base}/api/generate",
                json={"model": self._model, "keep_alive": OLLAMA_KEEP_ALIVE},
                timeout=REQUEST_TIMEOUT_S,
            )
            return response.status_code < 400
        except httpx.HTTPError:
            return False

    def schedule_pin(self) -> None:
        """Re-pin after a generation, without making the user wait for it.

        Each /v1 request resets Ollama's timer to its five-minute default, so
        the pin has to be renewed or it is undone by the very requests it
        exists to speed up. Fire and forget: this must never delay a response
        or fail one.
        """
        if not self._local:
            return
        try:
            task = asyncio.create_task(self.pin())
        except RuntimeError:
            return  # no running loop, e.g. under a synchronous test
        # Hold a reference until it finishes, or the loop may garbage-collect
        # the task mid-flight, and swallow the result either way.
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    def _reasoning_effort(self) -> str | None:
        """Cap reasoning depth for GPT-OSS models.

        Groq exposes reasoning_effort only for this model family (other
        models use an unrelated reasoning_format parameter instead). Without
        a cap, the model can spend the full output-token budget reasoning and
        never reach visible content, which is otherwise indistinguishable
        from an upstream failure.
        """
        return "low" if "gpt-oss" in self._model else None

    def _payload(self, request: GenerationRequest, stream: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self._model,
            "messages": build_messages(request),
            "max_tokens": MAX_OUTPUT_TOKENS,
            # Deterministic decoding: /score-style reproducibility means the same
            # question over the same context must not drift between runs.
            "temperature": 0.0,
            "stream": stream,
        }
        effort = self._reasoning_effort()
        if effort:
            payload["reasoning_effort"] = effort
        return payload

    async def warmup(self) -> bool:
        """Open a TLS connection ahead of the first real request.

        The first HTTPS call to a new host pays DNS plus handshake. Doing that
        during startup instead of during a user's first question removes a large
        one-off spike from the measured latency.
        """
        if self._local:
            # There is no TLS handshake to amortize against localhost, but the
            # weights still have to reach VRAM, and paying that on the first
            # user's question charges a one-off cost to steady-state latency.
            return await self.pin()
        if not self._keys.configured:
            return False
        try:
            await self._client.get("https://api.groq.com/openai/v1/models",
                                   headers=self._headers(), timeout=CONNECT_TIMEOUT_S)
            return True
        except httpx.HTTPError:
            return False

    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Yield answer text incrementally as the model produces it."""
        self._breaker.check()
        headers = self._headers()
        payload = self._payload(request, stream=True)

        started = time.perf_counter()
        first_token_at: float | None = None

        try:
            async with self._client.stream(
                "POST", self._api_url, headers=headers, json=payload
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")
                    if self._rotate_on(response.status_code):
                        async for piece in self.stream(request):
                            yield piece
                        return
                    self._breaker.record_failure()
                    raise UpstreamError(
                        f"Generation upstream returned {response.status_code}.",
                        detail=body[:300],
                    )

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        # A malformed frame mid-stream should not kill an answer
                        # that is otherwise arriving fine.
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    piece = (choices[0].get("delta") or {}).get("content")
                    if not piece:
                        continue
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    yield piece

        except httpx.TimeoutException as error:
            self._breaker.record_failure()
            raise UpstreamTimeoutError(
                "Generation upstream timed out.",
                detail=f"exceeded {REQUEST_TIMEOUT_S}s",
            ) from error
        except httpx.HTTPError as error:
            self._breaker.record_failure()
            raise UpstreamError(
                "Generation upstream is unreachable.", detail=str(error), retryable=True
            ) from error

        if first_token_at is None:
            self._breaker.record_failure()
            raise UpstreamError("Generation upstream returned no content.")
        self._breaker.record_success()

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Collect a full answer, reporting TTFT and whether it met the budget."""
        started = time.perf_counter()
        first_token_at: float | None = None
        pieces: list[str] = []

        attempt = 0
        while True:
            try:
                async for piece in self.stream(request):
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    pieces.append(piece)
                break
            except UpstreamError as error:
                # Retry only a connection-level fault, and only if nothing has
                # been emitted yet. An upstream that answered with an error
                # status will answer the same way again, and a timeout has
                # already spent the budget.
                if not error.retryable or attempt >= MAX_RETRIES or pieces:
                    raise
                attempt += 1
                pieces.clear()
                first_token_at = None

        total_ms = (time.perf_counter() - started) * 1000
        ttft_ms = ((first_token_at or time.perf_counter()) - started) * 1000
        text = "".join(pieces).strip()

        return GenerationResult(
            text=text,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            model=self._model,
            tokens=len(pieces),
            within_budget=ttft_ms <= TTFT_BUDGET_MS,
            model_refused=is_ungrounded_reply(text),
        )

    @property
    def tools(self) -> ToolRegistry:
        return self._tools

    async def _complete(self, messages: list[dict[str, Any]], use_tools: bool) -> dict[str, Any]:
        """One non-streaming completion, returning the raw assistant message.

        Tool rounds are non-streaming on purpose: a tool call has no partial
        form worth showing, and the answer that follows is streamed separately.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0.0,
            "stream": False,
        }
        effort = self._reasoning_effort()
        if effort:
            payload["reasoning_effort"] = effort
        if use_tools and len(self._tools):
            payload["tools"] = self._tools.schemas()
            payload["tool_choice"] = "auto"

        try:
            response = await self._client.post(self._api_url, headers=self._headers(), json=payload)
        except httpx.TimeoutException as error:
            self._breaker.record_failure()
            raise UpstreamTimeoutError(
                "Generation upstream timed out.", detail=f"exceeded {REQUEST_TIMEOUT_S}s"
            ) from error
        except httpx.HTTPError as error:
            self._breaker.record_failure()
            raise UpstreamError(
                "Generation upstream is unreachable.", detail=str(error), retryable=True
            ) from error

        if response.status_code >= 400:
            if self._rotate_on(response.status_code):
                return await self._complete(messages, use_tools)
            self._breaker.record_failure()
            raise UpstreamError(
                f"Generation upstream returned {response.status_code}.",
                detail=response.text[:300],
            )

        try:
            body = response.json()
        except ValueError as error:
            self._breaker.record_failure()
            raise UpstreamError("Generation upstream returned a non-JSON body.") from error

        choices = body.get("choices") or []
        if not choices:
            self._breaker.record_failure()
            raise UpstreamError("Generation upstream returned no choices.")

        self._breaker.record_success()
        return dict(choices[0].get("message") or {})

    async def generate_with_tools(self, request: GenerationRequest) -> GenerationResult:
        """Generate, letting the model call registered tools first.

        Runs model -> tools -> model until the model stops asking for tools or
        MAX_TOOL_ROUNDS is reached. Calls within a single round are executed
        concurrently, since they are independent by construction: the model
        issued them together without seeing any of their results.
        """
        self._breaker.check()
        started = time.perf_counter()

        messages: list[dict[str, Any]] = list(build_messages(request))
        collected: list[ToolResult] = []
        rounds = 0
        answer: str | None = None

        while rounds < MAX_TOOL_ROUNDS:
            message = await self._complete(messages, use_tools=True)
            calls = parse_tool_calls(message)
            if not calls:
                # The model answered instead of calling a tool. That reply IS
                # the answer; asking again would spend another round trip to
                # receive the same thing.
                answer = str(message.get("content") or "").strip()
                break

            rounds += 1
            messages.append(message)
            # Calls issued together are independent by construction: the model
            # chose them without seeing any of their results.
            results = await asyncio.gather(*(self._tools.execute(call) for call in calls))
            collected.extend(results)
            messages.extend(result.to_message() for result in results)

        if answer is None:
            # Only reachable at the round cap: force prose by withholding tools,
            # so a model stuck in a call loop still returns something usable.
            final = await self._complete(messages, use_tools=False)
            answer = str(final.get("content") or "").strip()

        total_ms = (time.perf_counter() - started) * 1000
        return GenerationResult(
            text=answer,
            ttft_ms=total_ms,  # non-streaming: the first token arrives with the last
            total_ms=total_ms,
            model=self._model,
            tokens=len(answer.split()),
            within_budget=total_ms <= TTFT_BUDGET_MS,
            model_refused=is_ungrounded_reply(answer),
            tool_results=collected,
            tool_rounds=rounds,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
