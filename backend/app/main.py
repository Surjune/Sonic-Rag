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
import uuid
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Deque

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.config import (
    CORS_ORIGINS,
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
)
from app.harness import GenerationRequest, GroqHarness
from app.retrieval import Hit, engine
from app.stt_service import SarvamSttService
from app.telemetry import LatencyTrace
from app.translation import SarvamTranslator, detect_script

# Recent guardrail decisions, surfaced by the audit-log view. Bounded so a long
# running demo cannot grow memory without limit.
AUDIT_LOG: Deque[AuditEntry] = deque(maxlen=200)

harness: GroqHarness
stt: SarvamSttService
translator: SarvamTranslator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load everything expensive once, before the first request arrives."""
    global harness, stt, translator
    harness = GroqHarness()
    stt = SarvamSttService()
    translator = SarvamTranslator()

    try:
        # Blocking disk and ONNX work; keep it off the event loop.
        await asyncio.to_thread(engine.load)
    except IndexNotLoadedError:
        # Start anyway so /health can report the problem instead of the
        # container crash-looping with the reason buried in logs.
        pass

    await harness.warmup()
    yield

    await harness.aclose()
    await stt.aclose()
    await translator.aclose()


app = FastAPI(title="Sonic-RAG", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
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
        "groq_model": GROQ_MODEL,
        "circuit": harness.circuit_state.value,
        "sarvam_configured": stt.configured,
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

    with trace.stage("llm"):
        result = await harness.generate(
            GenerationRequest(query=english_query, contexts=contexts, language=answer_language)
        )
    trace.record("llm_ttft", round(result.ttft_ms, 3))

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
        "within_budget": result.within_budget,
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
        try:
            async for piece in harness.stream(generation):
                yield f"event: token\ndata: {json.dumps({'t': piece})}\n\n"
        except SonicRagError as error:
            yield f"event: error\ndata: {json.dumps(error.to_dict())}\n\n"
            return

        yield f"event: done\ndata: {json.dumps({'latency': trace.as_dict(), 'grounded': True})}\n\n"

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
) -> Any:
    """Voice question in. saaras returns English directly, so no translate hop."""
    trace = LatencyTrace()
    audio = await file.read()

    with trace.stage("stt"):
        transcription = await stt.transcribe(audio, filename=file.filename or "audio.wav")

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

    hits, scores = await _retrieve(verdict.normalized_query, top_k, trace)

    with trace.stage("guardrail_grounding"):
        grounding = check_grounding(scores)
    AUDIT_LOG.appendleft(audit_grounding(grounding, verdict.normalized_query))

    transcript = {
        "native": transcription.native_text,
        "english": transcription.english_text,
        "detected_language": transcription.detected_language_code,
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
    with trace.stage("llm"):
        result = await harness.generate(
            GenerationRequest(
                query=verdict.normalized_query, contexts=contexts, language=answer_language
            )
        )
    trace.record("llm_ttft", round(result.ttft_ms, 3))

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
        "within_budget": result.within_budget,
        "latency": trace.as_dict(),
    }


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


@app.get("/api/stats")
async def stats() -> dict[str, Any]:
    """Index composition, for the chunking explorer view."""
    return {
        "index_size": engine.size,
        "meta": engine.meta,
        "threshold": SIMILARITY_THRESHOLD,
        "supported_languages": list(SUPPORTED_LANGS),
    }
