"""FastAPI orchestration for Sonic-RAG.

Request flow, with every stage timed independently:

    input guardrail -> [translate if typed Indic] -> embed -> FAISS
        -> grounding guardrail -> Groq

Both guardrails can short-circuit. A blocked or ungrounded request never reaches
the model, which keeps the refusal path near-instant and spends no tokens.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Deque

import httpx
from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.config import (
    BACKEND_DIR,
    CONNECT_TIMEOUT_S,
    CORS_ORIGINS,
    OLLAMA_API_URL,
    OLLAMA_MODEL,
    DEFAULT_TOP_K,
    GROQ_MODEL,
    SIMILARITY_THRESHOLD,
    SUPPORTED_LANGS,
    TTFT_BUDGET_MS,
)
from app.exceptions import IndexNotLoadedError, SonicRagError
from app.guardrails import (
    AuditEntry,
    audit_grounding,
    audit_input,
    check_grounding,
    check_input,
    detect_small_talk,
    is_probably_silence,
    NO_SPEECH_MESSAGE,
)
from app.chunk_eval import load_cached
from app.chunkers import STRATEGY_ORDER, get_chunker
from app.harness import GenerationRequest, GroqHarness, is_ungrounded_reply
from app.retrieval import Hit, engine
from app.stt_service import SpeechToText
from app.tts_service import TextToSpeech
from app.telemetry import LatencyTrace
from app.tools import Tool, ToolRegistry
from app.translation import SarvamTranslator, detect_script

# Recent guardrail decisions, surfaced by the audit-log view. Bounded so a long
# running demo cannot grow memory without limit.
AUDIT_LOG: Deque[AuditEntry] = deque(maxlen=200)

harness: GroqHarness
local_harness: GroqHarness
stt: SpeechToText
translator: SarvamTranslator
tts: TextToSpeech


def build_tool_registry() -> ToolRegistry:
    """Tools the model may call.

    Registered here rather than in the harness so the harness stays generic:
    the API layer is the only place permitted to reach into retrieval.
    """

    async def search_corpus(query: str, top_k: int = 3) -> dict[str, Any]:
        """Let the model run its own retrieval when the given context is thin."""
        if not engine.ready:
            return {"error": "index not loaded"}
        bounded = max(1, min(int(top_k), 10))
        vector = await asyncio.to_thread(engine.embed_query, str(query))
        hits = engine.search(vector, bounded)
        return {
            "query": query,
            "results": [
                {"score": round(hit.score, 4), "text": hit.parent_english[:400]} for hit in hits
            ],
        }

    async def index_stats() -> dict[str, Any]:
        """Answer questions about the index without inventing figures."""
        return {
            "vectors": engine.size,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "languages": list(SUPPORTED_LANGS),
            **{k: v for k, v in engine.meta.items() if k in {"model", "dim", "vector_space"}},
        }

    return ToolRegistry(
        [
            Tool(
                name="search_corpus",
                description=(
                    "Search the indexed corpus for passages relevant to a question. "
                    "Use when the provided context does not cover the question."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "search query in English"},
                        "top_k": {
                            "type": "integer",
                            "description": "how many passages to return (1-10)",
                            "default": 3,
                        },
                    },
                    "required": ["query"],
                },
                handler=search_corpus,
            ),
            Tool(
                name="index_stats",
                description="Return facts about the retrieval index: size, model, languages.",
                parameters={"type": "object", "properties": {}},
                handler=index_stats,
            ),
        ]
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load everything expensive once, before the first request arrives."""
    global harness, local_harness, stt, translator, tts
    tools = build_tool_registry()
    harness = GroqHarness(tools=tools)
    # Built regardless of whether Ollama is installed. Constructing it costs
    # nothing -- no connection is opened until a request selects it -- and
    # having it ready is what lets the interface offer the switch instead of
    # requiring a restart to change backends.
    local_harness = GroqHarness(provider="ollama", tools=tools)
    stt = SpeechToText()
    translator = SarvamTranslator()
    tts = TextToSpeech()

    try:
        # Blocking disk and ONNX work; keep it off the event loop.
        await asyncio.to_thread(engine.load)
    except IndexNotLoadedError:
        # Start anyway so /health can report the problem instead of the
        # container crash-looping with the reason buried in logs.
        pass

    # Pre-connect every upstream concurrently. The first HTTPS call to each host
    # pays DNS plus a TLS handshake; paying it here keeps it out of the first
    # user's measured latency.
    await asyncio.gather(
        harness.warmup(),
        stt.warmup(),
        translator.warmup(),
        tts.warmup(),
        # Wakes Ollama and pulls the model into VRAM if it happens to be
        # running. Failure is the normal case -- most people will not have it
        # installed -- so it is gathered with the rest and its result ignored.
        local_harness.warmup(),
        return_exceptions=True,
    )
    yield

    await harness.aclose()
    await local_harness.aclose()
    await stt.aclose()
    await translator.aclose()
    await tts.aclose()


app = FastAPI(title="Sonic-RAG", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def attach_request_id(request: Request, call_next: Any) -> Any:
    """Give every request a correlation id, echoed back on the response."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(SonicRagError)
async def handle_domain_error(request: Request, error: SonicRagError) -> JSONResponse:
    """Map typed errors onto a consistent envelope, never a stack trace."""
    return JSONResponse(
        status_code=error.status,
        content={"error": error.to_dict(), "request_id": getattr(request.state, "request_id", "")},
    )


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    language: str | None = Field(default=None, description="answer language; auto-detected if unset")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)
    generate: bool = Field(
        default=True,
        description="set false to run retrieval only; isolates local pipeline latency "
        "from the network-bound model call",
    )
    use_tools: bool = Field(
        default=False,
        description="let the model call registered tools before answering; costs extra "
        "round trips, so it is opt-in rather than the default path",
    )
    provider: str | None = Field(
        default=None,
        description='generation backend: "groq" (default) or "ollama" for a model '
        "running on the caller's own machine. Selected per request so the choice "
        "can be a switch in the interface rather than a restart.",
    )


def select_harness(provider: str | None) -> GroqHarness:
    """Pick the generation backend for one request.

    Anything unrecognised falls back to the default rather than erroring: a bad
    value in a query string should not take the answer away from the user.
    """
    return local_harness if (provider or "").strip().lower() == "ollama" else harness


def _serialize_hits(hits: list[Hit], language: str) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": hit.chunk_id,
            "score": round(hit.score, 4),
            "text_english": hit.text_english,
            "display_text": hit.display_text(language),
            "is_selected": hit.is_selected,
            "above_threshold": hit.score >= SIMILARITY_THRESHOLD,
        }
        for hit in hits
    ]


async def _retrieve(
    query_text: str, top_k: int, trace: LatencyTrace
) -> tuple[list[Hit], list[float]]:
    """Embed and search. Embedding runs off the event loop; FAISS stays inline."""
    with trace.stage("embed"):
        vector = await asyncio.to_thread(engine.embed_query, query_text)
    with trace.stage("faiss"):
        hits = engine.search(vector, top_k)
    return hits, [hit.score for hit in hits]


async def _prepare(
    raw_query: str, language: str | None, trace: LatencyTrace
) -> tuple[str, str, str] | JSONResponse:
    """Guardrail, then translate a typed Indic query into the English space.

    Returns (english_query, display_query, answer_language) or a refusal.
    """
    with trace.stage("guardrail_input"):
        verdict = check_input(raw_query)
    AUDIT_LOG.appendleft(audit_input(verdict, raw_query))

    if not verdict.allowed:
        return JSONResponse(
            status_code=400,
            content={
                "blocked": True,
                "stage": "input",
                "code": verdict.code,
                "message": verdict.description,
                "matched": verdict.matched_text,
                "latency": trace.as_dict(),
            },
        )

    detected = detect_script(verdict.normalized_query)
    answer_language = language if language in SUPPORTED_LANGS else detected

    # A greeting is not a question, and sending it down the retrieval path
    # produces a confident-looking refusal to "hi" after a full model round
    # trip. Answered here it costs a regex match, no translation, no embedding
    # and no tokens. Not a block: nothing is wrong with the input, it simply
    # has a better answer than retrieval can give.
    small_talk = detect_small_talk(verdict.normalized_query, answer_language)
    if small_talk is not None:
        kind, reply = small_talk
        return JSONResponse(
            status_code=200,
            content={
                "answer": reply,
                # Grounded and unblocked on purpose: the interface should show
                # this as an ordinary answer, not as a refusal or an error.
                "grounded": True,
                "blocked": False,
                "generated": False,
                "small_talk": kind,
                "language": answer_language,
                "query": {"raw": verdict.normalized_query, "english": verdict.normalized_query},
                "contexts": [],
                "latency": trace.as_dict(),
            },
        )

    with trace.stage("translate"):
        translation = await translator.to_english(verdict.normalized_query, detected)

    return translation.text, verdict.normalized_query, answer_language


@app.get("/health")
async def health() -> dict[str, Any]:
    """Readiness plus what is actually configured, for debugging a deploy."""
    return {
        "status": "ok" if engine.ready else "degraded",
        "index_loaded": engine.ready,
        "index_size": engine.size,
        "index_meta": engine.meta,
        "groq_configured": harness.configured,
        # Which generation backend is actually serving, not which one is
        # configured in the file -- a silent provider switch is a debugging trap.
        "llm_provider": harness.provider,
        "groq_model": harness.model,
        # How many keys are loaded and which is active. Labels only, never
        # the keys themselves.
        "groq_key": harness.key_label,
        "circuit": harness.circuit_state.value,
        "stt_configured": stt.configured,
        "stt_providers": stt.providers,
        "tts_configured": tts.configured,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "ttft_budget_ms": TTFT_BUDGET_MS,
    }


@app.post("/api/query")
async def query(request: QueryRequest) -> Any:
    """Text question in, grounded answer out, with per-stage timings."""
    trace = LatencyTrace()

    prepared = await _prepare(request.query, request.language, trace)
    if isinstance(prepared, JSONResponse):
        return prepared
    english_query, display_query, answer_language = prepared

    hits, scores = await _retrieve(english_query, request.top_k, trace)

    with trace.stage("guardrail_grounding"):
        grounding = check_grounding(scores)
    AUDIT_LOG.appendleft(audit_grounding(grounding, english_query))

    if not grounding.allowed:
        # Refused without calling the model: no tokens spent, no invented answer.
        return {
            "answer": grounding.message,
            "grounded": False,
            "blocked": True,
            "stage": "grounding",
            "code": grounding.code,
            "language": answer_language,
            "query": {"raw": display_query, "english": english_query},
            "top_score": round(grounding.top_score, 4),
            "threshold": grounding.threshold,
            "contexts": _serialize_hits(hits, answer_language),
            "latency": trace.as_dict(),
        }

    contexts = engine.build_contexts(hits)

    if not request.generate:
        # Retrieval-only: everything above is local CPU work, so this measures
        # the pipeline latency that is actually under our control.
        return {
            "answer": "",
            "grounded": True,
            "blocked": False,
            "generated": False,
            "language": answer_language,
            "query": {"raw": display_query, "english": english_query},
            "top_score": round(grounding.top_score, 4),
            "threshold": grounding.threshold,
            "contexts": _serialize_hits(hits, answer_language),
            "latency": trace.as_dict(),
        }

    generation = GenerationRequest(
        query=english_query, contexts=contexts, language=answer_language
    )
    backend = select_harness(request.provider)
    llm_started = time.perf_counter()
    with trace.stage("llm"):
        result = await (
            backend.generate_with_tools(generation) if request.use_tools
            else backend.generate(generation)
        )
    trace.record("llm_ttft", round(result.ttft_ms, 3))
    trace.mark("first_output", llm_started + result.ttft_ms / 1000)
    # Renew the local model's residency. Each generation resets Ollama's timer
    # to its five-minute default, so without this the next question after a
    # pause pays a full reload.
    backend.schedule_pin()

    return {
        "answer": result.text,
        "generated": True,
        # Retrieval and the model judge groundedness independently. When the
        # model declines despite the score clearing the threshold, saying
        # "grounded" would contradict the answer shown next to it.
        "grounded": not result.model_refused,
        "model_refused": result.model_refused,
        "code": "MODEL_UNGROUNDED" if result.model_refused else None,
        "blocked": False,
        "language": answer_language,
        "query": {"raw": display_query, "english": english_query},
        "top_score": round(grounding.top_score, 4),
        "threshold": grounding.threshold,
        "contexts": _serialize_hits(hits, answer_language),
        "model": result.model,
        "provider": backend.provider,
        "within_budget": result.within_budget,
        "tool_rounds": result.tool_rounds,
        "tool_calls": [
            {
                "name": call.name,
                "ok": call.ok,
                "latency_ms": round(call.latency_ms, 3),
                "content": call.content[:400],
            }
            for call in result.tool_results
        ],
        "latency": trace.as_dict(),
    }


@app.post("/api/query/stream")
async def query_stream(request: QueryRequest) -> StreamingResponse:
    """Same pipeline, streamed as SSE so the UI can render the first token."""

    async def events() -> AsyncIterator[str]:
        trace = LatencyTrace()

        prepared = await _prepare(request.query, request.language, trace)
        if isinstance(prepared, JSONResponse):
            body = json.loads(bytes(prepared.body).decode())
            # Small talk is an answer, not a refusal, so it is streamed like
            # one. Emitting it as `blocked` would paint a greeting red.
            if body.get("small_talk"):
                yield f"event: meta\ndata: {json.dumps(body)}\n\n"
                yield f"event: token\ndata: {json.dumps({'t': body['answer']})}\n\n"
                done = {"latency": body["latency"], "grounded": True, "model_refused": False}
                yield f"event: done\ndata: {json.dumps(done)}\n\n"
                return
            yield f"event: blocked\ndata: {json.dumps(body)}\n\n"
            return
        english_query, display_query, answer_language = prepared

        hits, scores = await _retrieve(english_query, request.top_k, trace)

        with trace.stage("guardrail_grounding"):
            grounding = check_grounding(scores)
        AUDIT_LOG.appendleft(audit_grounding(grounding, english_query))

        meta = {
            "language": answer_language,
            "query": {"raw": display_query, "english": english_query},
            "contexts": _serialize_hits(hits, answer_language),
            "top_score": round(grounding.top_score, 4),
            "threshold": grounding.threshold,
            "latency": trace.as_dict(),
        }
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

        if not grounding.allowed:
            payload = {"answer": grounding.message, "code": grounding.code, "grounded": False}
            yield f"event: blocked\ndata: {json.dumps(payload)}\n\n"
            return

        contexts = engine.build_contexts(hits)
        generation = GenerationRequest(
            query=english_query, contexts=contexts, language=answer_language
        )
        backend = select_harness(request.provider)
        pieces: list[str] = []
        first_token_at: float | None = None
        llm_started = time.perf_counter()
        try:
            with trace.stage("llm"):
                async for piece in backend.stream(generation):
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    pieces.append(piece)
                    yield f"event: token\ndata: {json.dumps({'t': piece})}\n\n"
        except SonicRagError as error:
            yield f"event: error\ndata: {json.dumps(error.to_dict())}\n\n"
            return

        if first_token_at is not None:
            trace.record("llm_ttft", round((first_token_at - llm_started) * 1000, 3))
            # When the user could start reading. This is the latency figure;
            # everything after it is the answer still arriving, which the
            # reader is already consuming rather than waiting on.
            trace.mark("first_output", first_token_at)
        # The model is a second, independent judge of groundedness: a passage
        # can clear the cosine threshold while the model still finds it
        # unusable. Checking its actual reply, not just assuming success once
        # streaming completes, is what makes "grounded" here true rather than
        # a default.
        backend.schedule_pin()
        model_refused = is_ungrounded_reply("".join(pieces))
        yield (
            "event: done\ndata: "
            f"{json.dumps({'latency': trace.as_dict(), 'grounded': not model_refused, 'model_refused': model_refused})}"
            "\n\n"
        )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/voice")
async def voice(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    top_k: int = Form(default=DEFAULT_TOP_K),
    provider: str | None = Form(default=None),
) -> Any:
    """Voice question in. saaras returns English directly, so no translate hop."""
    trace = LatencyTrace()
    audio = await file.read()

    with trace.stage("stt"):
        # The caller's language choice is a stronger signal than the fallback
        # provider's guess, which mishears Hindi as Urdu.
        transcription = await stt.transcribe(
            audio,
            filename=file.filename or "audio.wav",
            language_hint=language if language in SUPPORTED_LANGS else None,
        )

    answer_language = language if language in SUPPORTED_LANGS else transcription.language

    with trace.stage("guardrail_input"):
        verdict = check_input(transcription.english_text)
    AUDIT_LOG.appendleft(audit_input(verdict, transcription.english_text))

    if not verdict.allowed:
        return JSONResponse(
            status_code=400,
            content={
                "blocked": True,
                "stage": "input",
                "code": verdict.code,
                "message": verdict.description,
                "transcript": {
                    "native": transcription.native_text,
                    "english": transcription.english_text,
                },
                "latency": trace.as_dict(),
            },
        )

    # Nothing was said. Every layer below this would behave correctly on the
    # artifact a speech model returns for silence, and correctly produce an
    # answer to a question nobody asked.
    if is_probably_silence(transcription.english_text):
        return {
            "answer": NO_SPEECH_MESSAGE.get(answer_language, NO_SPEECH_MESSAGE["en"]),
            "grounded": False,
            "blocked": True,
            "stage": "input",
            "code": "NO_SPEECH",
            "generated": False,
            "language": answer_language,
            "transcript": {
                "native": transcription.native_text,
                "english": transcription.english_text,
                "detected_language": transcription.detected_language_code,
                "provider": transcription.provider,
                "fallback_reason": transcription.fallback_reason,
            },
            "contexts": [],
            "latency": trace.as_dict(),
        }

    # A spoken "hello" deserves the same treatment as a typed one.
    small_talk = detect_small_talk(verdict.normalized_query, answer_language)
    if small_talk is not None:
        kind, reply = small_talk
        return {
            "answer": reply,
            "grounded": True,
            "blocked": False,
            "generated": False,
            "small_talk": kind,
            "language": answer_language,
            "transcript": {
                "native": transcription.native_text,
                "english": transcription.english_text,
                "detected_language": transcription.detected_language_code,
                "provider": transcription.provider,
                "fallback_reason": transcription.fallback_reason,
            },
            "contexts": [],
            "latency": trace.as_dict(),
        }

    hits, scores = await _retrieve(verdict.normalized_query, top_k, trace)

    with trace.stage("guardrail_grounding"):
        grounding = check_grounding(scores)
    AUDIT_LOG.appendleft(audit_grounding(grounding, verdict.normalized_query))

    # Which provider answered is part of the result, not a hidden detail: a
    # silent failover nobody can see is a debugging trap.
    trace.record("stt_provider_latency", round(transcription.latency_ms, 3))
    transcript = {
        "native": transcription.native_text,
        "english": transcription.english_text,
        "detected_language": transcription.detected_language_code,
        "provider": transcription.provider,
        "fallback_reason": transcription.fallback_reason,
    }

    if not grounding.allowed:
        return {
            "answer": grounding.message,
            "grounded": False,
            "blocked": True,
            "stage": "grounding",
            "code": grounding.code,
            "language": answer_language,
            "transcript": transcript,
            "top_score": round(grounding.top_score, 4),
            "contexts": _serialize_hits(hits, answer_language),
            "latency": trace.as_dict(),
        }

    contexts = engine.build_contexts(hits)
    backend = select_harness(provider)
    llm_started = time.perf_counter()
    with trace.stage("llm"):
        result = await backend.generate(
            GenerationRequest(
                query=verdict.normalized_query, contexts=contexts, language=answer_language
            )
        )
    trace.record("llm_ttft", round(result.ttft_ms, 3))
    trace.mark("first_output", llm_started + result.ttft_ms / 1000)
    backend.schedule_pin()

    return {
        "answer": result.text,
        "grounded": not result.model_refused,
        "model_refused": result.model_refused,
        "code": "MODEL_UNGROUNDED" if result.model_refused else None,
        "blocked": False,
        "language": answer_language,
        "transcript": transcript,
        "top_score": round(grounding.top_score, 4),
        "contexts": _serialize_hits(hits, answer_language),
        "model": result.model,
        "provider": backend.provider,
        "within_budget": result.within_budget,
        "latency": trace.as_dict(),
    }


@app.post("/api/voice/stream")
async def voice_stream(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    top_k: int = Form(default=DEFAULT_TOP_K),
    provider: str | None = Form(default=None),
) -> StreamingResponse:
    """Voice in, answer streamed out.

    The non-streaming /api/voice waits for the whole answer before replying, so
    the user stares at nothing for the full generation. Here the transcript is
    emitted the moment speech recognition returns, context as soon as retrieval
    finishes, and tokens as they arrive -- so the first readable output lands at
    time-to-first-token rather than at completion.
    """
    # Read the upload before the generator starts: the request body is not
    # guaranteed to still be readable once streaming has begun.
    audio = await file.read()
    filename = file.filename or "audio.wav"

    async def events() -> AsyncIterator[str]:
        trace = LatencyTrace()

        try:
            with trace.stage("stt"):
                transcription = await stt.transcribe(
                    audio,
                    filename=filename,
                    language_hint=language if language in SUPPORTED_LANGS else None,
                )
        except SonicRagError as error:
            yield f"event: error\ndata: {json.dumps(error.to_dict())}\n\n"
            return

        answer_language = (
            language if language in SUPPORTED_LANGS else transcription.language
        )
        transcript = {
            "native": transcription.native_text,
            "english": transcription.english_text,
            "detected_language": transcription.detected_language_code,
            "provider": transcription.provider,
            "fallback_reason": transcription.fallback_reason,
        }
        # Show the user their own words immediately, well before the answer.
        yield f"event: transcript\ndata: {json.dumps({'transcript': transcript, 'language': answer_language, 'latency': trace.as_dict()})}\n\n"

        with trace.stage("guardrail_input"):
            verdict = check_input(transcription.english_text)
        AUDIT_LOG.appendleft(audit_input(verdict, transcription.english_text))

        if not verdict.allowed:
            payload = {
                "blocked": True,
                "stage": "input",
                "code": verdict.code,
                "message": verdict.description,
                "latency": trace.as_dict(),
            }
            yield f"event: blocked\ndata: {json.dumps(payload)}\n\n"
            return

        if is_probably_silence(transcription.english_text):
            payload = {
                "blocked": True,
                "stage": "input",
                "code": "NO_SPEECH",
                "message": NO_SPEECH_MESSAGE.get(answer_language, NO_SPEECH_MESSAGE["en"]),
                "answer": NO_SPEECH_MESSAGE.get(answer_language, NO_SPEECH_MESSAGE["en"]),
                "grounded": False,
                "latency": trace.as_dict(),
            }
            yield f"event: blocked\ndata: {json.dumps(payload)}\n\n"
            return

        small_talk = detect_small_talk(verdict.normalized_query, answer_language)
        if small_talk is not None:
            kind, reply = small_talk
            meta = {
                "language": answer_language,
                "transcript": transcript,
                "contexts": [],
                "small_talk": kind,
                "latency": trace.as_dict(),
            }
            yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
            yield f"event: token\ndata: {json.dumps({'t': reply})}\n\n"
            done = {"latency": trace.as_dict(), "grounded": True, "model_refused": False}
            yield f"event: done\ndata: {json.dumps(done)}\n\n"
            return

        hits, scores = await _retrieve(verdict.normalized_query, top_k, trace)

        with trace.stage("guardrail_grounding"):
            grounding = check_grounding(scores)
        AUDIT_LOG.appendleft(audit_grounding(grounding, verdict.normalized_query))

        meta = {
            "language": answer_language,
            "transcript": transcript,
            "contexts": _serialize_hits(hits, answer_language),
            "top_score": round(grounding.top_score, 4),
            "threshold": grounding.threshold,
            "latency": trace.as_dict(),
        }
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

        if not grounding.allowed:
            payload = {"answer": grounding.message, "code": grounding.code, "grounded": False}
            yield f"event: blocked\ndata: {json.dumps(payload)}\n\n"
            return

        generation = GenerationRequest(
            query=verdict.normalized_query,
            contexts=engine.build_contexts(hits),
            language=answer_language,
        )
        backend = select_harness(provider)
        pieces: list[str] = []
        first_token_at: float | None = None
        llm_started = time.perf_counter()
        try:
            with trace.stage("llm"):
                async for piece in backend.stream(generation):
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    pieces.append(piece)
                    yield f"event: token\ndata: {json.dumps({'t': piece})}\n\n"
        except SonicRagError as error:
            yield f"event: error\ndata: {json.dumps(error.to_dict())}\n\n"
            return

        if first_token_at is not None:
            trace.record("llm_ttft", round((first_token_at - llm_started) * 1000, 3))
            # When the user could start reading. This is the latency figure;
            # everything after it is the answer still arriving, which the
            # reader is already consuming rather than waiting on.
            trace.mark("first_output", first_token_at)
        backend.schedule_pin()
        model_refused = is_ungrounded_reply("".join(pieces))
        yield (
            "event: done\ndata: "
            f"{json.dumps({'latency': trace.as_dict(), 'grounded': not model_refused, 'model_refused': model_refused})}"
            "\n\n"
        )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/audit")
async def audit(limit: int = 50) -> dict[str, Any]:
    """Recent guardrail decisions for the audit-log view."""
    entries = list(AUDIT_LOG)[: max(1, min(limit, AUDIT_LOG.maxlen or 200))]
    return {
        "entries": [
            {
                "stage": entry.stage,
                "code": entry.code,
                "allowed": entry.allowed,
                "latency_ms": round(entry.latency_ms, 4),
                "detail": entry.detail,
                "query_preview": entry.query_preview,
            }
            for entry in entries
        ],
        "blocked_count": sum(1 for entry in AUDIT_LOG if not entry.allowed),
        "total_count": len(AUDIT_LOG),
    }


class PreviewRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    lang: str = Field(default="en")
    strategies: list[str] = Field(default_factory=lambda: list(STRATEGY_ORDER))


@app.post("/api/chunking/preview")
async def chunking_preview(request: PreviewRequest) -> dict[str, Any]:
    """Run each strategy over one passage and return the chunks it produces.

    Pure CPU string work with no embedding, so the same text can be compared
    across strategies instantly and at no cost.
    """
    requested = [name for name in request.strategies if name in STRATEGY_ORDER] or list(
        STRATEGY_ORDER
    )
    trace = LatencyTrace()
    output: list[dict[str, Any]] = []

    for name in requested:
        chunker = get_chunker(name)
        with trace.stage(name):
            chunks = chunker.chunk(request.text, lang=request.lang)
        sizes = [chunk.length for chunk in chunks]
        output.append(
            {
                "strategy": name,
                "description": chunker.description,
                "count": len(chunks),
                "mean_chars": round(sum(sizes) / len(sizes), 1) if sizes else 0,
                "min_chars": min(sizes) if sizes else 0,
                "max_chars": max(sizes) if sizes else 0,
                "chunks": [
                    {
                        "index": chunk.index,
                        "text": chunk.text,
                        "embed_text": chunk.embed_text,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                        "length": chunk.length,
                    }
                    for chunk in chunks
                ],
            }
        )

    return {"source_chars": len(request.text), "strategies": output, "latency": trace.as_dict()}


@app.get("/api/chunking/compare")
async def chunking_compare() -> dict[str, Any]:
    """Serve the cached strategy comparison.

    Computed offline by `python -m app.chunk_eval`: embedding several thousand
    chunks four times over takes minutes, which is far too slow to run inside a
    request. Absent results say so rather than returning invented numbers.
    """
    cached = load_cached()
    if cached is None:
        return {
            "available": False,
            "message": "No comparison has been computed yet.",
            "how_to_generate": "python -m app.chunk_eval --queries 40",
        }
    return {"available": True, **cached}


@app.get("/api/tools")
async def tools_manifest() -> dict[str, Any]:
    """What the model is allowed to call, for the interface to display."""
    return {"tools": harness.tools.schemas(), "names": harness.tools.names}


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    language: str = Field(default="en")


@app.post("/api/speak")
async def speak(request: SpeakRequest) -> dict[str, Any]:
    """Synthesize an answer so it can be heard as well as read.

    Separate from /api/query rather than folded into it. Synthesis costs a
    round trip and Sarvam quota, and most answers are read rather than played,
    so making every query pay for audio nobody asked for would be wrong. The
    interface calls this only when the user wants sound.
    """
    trace = LatencyTrace()
    language = request.language if request.language in SUPPORTED_LANGS else "en"

    with trace.stage("tts"):
        speech = await tts.speak(request.text, language)

    return {
        "audio_base64": speech.audio_base64,
        "format": "wav",
        "language": speech.language,
        "model": speech.model,
        "speaker": speech.speaker,
        "characters": speech.characters,
        "truncated": speech.truncated,
        "latency": trace.as_dict(),
    }


@app.get("/api/providers")
async def providers() -> dict[str, Any]:
    """Which generation backends this deployment can actually reach right now.

    Ollama is probed live rather than assumed from configuration: whether it is
    installed, running and has the model pulled are three different things, and
    an interface that offers a switch to a backend that cannot answer is worse
    than one that offers no switch. The install and pull commands travel with
    the negative answer so the interface can say what to do about it.
    """
    available: list[str] = []
    models: list[str] = []
    detail = ""

    base = OLLAMA_API_URL.split("/v1/")[0]
    try:
        response = await local_harness._client.get(f"{base}/api/tags", timeout=1.5)
        if response.status_code < 400:
            models = [str(m.get("name")) for m in (response.json().get("models") or [])]
            if any(m.split(":")[0] == OLLAMA_MODEL.split(":")[0] for m in models):
                available.append("ollama")
            else:
                detail = f"Ollama is running but {OLLAMA_MODEL} is not pulled."
        else:
            detail = f"Ollama responded {response.status_code}."
    except Exception:
        detail = "Ollama is not running on this machine."

    if harness.configured:
        available.insert(0, "groq")

    return {
        "default": harness.provider,
        "available": available,
        "groq": {
            "id": "groq",
            "label": "Groq",
            "model": harness.model,
            "ready": harness.configured,
            "local": False,
            "note": "Hosted LPU. No local install, but every call pays a network round trip.",
        },
        "ollama": {
            "id": "ollama",
            "label": "Local (Ollama)",
            "model": OLLAMA_MODEL,
            "ready": "ollama" in available,
            "local": True,
            "models_present": models,
            "detail": detail,
            "install_url": "https://ollama.com/download",
            "pull_command": f"ollama pull {OLLAMA_MODEL}",
            "note": (
                "Runs on your own machine. Measured here at 82ms TTFT against "
                "Groq's 438ms, because there is no network hop -- but it is a 3B "
                "model rather than a 20B one, and it needs a GPU to be quick."
            ),
        },
    }


@app.post("/api/providers/pull")
async def pull_local_model() -> StreamingResponse:
    """Download the local model, streaming progress so the wait is visible.

    A 2GB download behind a button with no feedback looks like a hang, and the
    honest thing is to show the bytes. Ollama's native pull API reports total
    and completed per layer, which is enough for a real progress bar rather
    than a spinner.

    This can only work when Ollama is reachable from the server, which means
    the two are on the same machine. On a hosted deployment they are not: the
    backend's localhost is the server, not the visitor's laptop, and the
    interface says so rather than offering a button that cannot work.
    """

    async def events() -> AsyncIterator[str]:
        base = OLLAMA_API_URL.split("/v1/")[0]
        try:
            async with local_harness._client.stream(
                "POST",
                f"{base}/api/pull",
                json={"model": OLLAMA_MODEL, "stream": True},
                timeout=httpx.Timeout(None, connect=CONNECT_TIMEOUT_S),
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")[:200]
                    yield f"event: error\ndata: {json.dumps({'message': body})}\n\n"
                    return

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        frame = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if frame.get("error"):
                        yield f"event: error\ndata: {json.dumps({'message': frame['error']})}\n\n"
                        return
                    total = int(frame.get("total") or 0)
                    completed = int(frame.get("completed") or 0)
                    payload = {
                        "status": frame.get("status", ""),
                        "total": total,
                        "completed": completed,
                        "percent": round(100 * completed / total, 1) if total else None,
                    }
                    yield f"event: progress\ndata: {json.dumps(payload)}\n\n"
        except httpx.HTTPError as error:
            yield f"event: error\ndata: {json.dumps({'message': str(error)})}\n\n"
            return

        # Pin it immediately: a model that was just downloaded should be ready
        # to answer, not evicted five minutes later having never been used.
        await local_harness.pin()
        yield f"event: done\ndata: {json.dumps({'model': OLLAMA_MODEL})}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/stats")
async def stats() -> dict[str, Any]:
    """Index composition, for the chunking explorer view."""
    return {
        "index_size": engine.size,
        "meta": engine.meta,
        "threshold": SIMILARITY_THRESHOLD,
        "supported_languages": list(SUPPORTED_LANGS),
    }


# If the frontend was built into dist (e.g. in the multi-stage Docker build),
# serve it from root so the container functions as an all-in-one deployment.
FRONTEND_DIST = BACKEND_DIR.parent / "frontend" / "dist"
if FRONTEND_DIST.exists() and (FRONTEND_DIST / "index.html").exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
