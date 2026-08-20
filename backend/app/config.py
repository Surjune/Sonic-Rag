"""Shared configuration and tuned constants for the Sonic-RAG pipeline.

Every value that affects retrieval quality or the latency budget lives here so
the indexer, the guardrails and the API cannot drift apart. A threshold changed
in one place and not the other would silently break grounding.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --- paths ------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[1]

# Load .env here, from a path anchored to this file, rather than relying on the
# caller to do it. A relative path in a launch command resolves against the
# working directory, so starting the server from the repo root instead of
# backend/ silently loaded nothing and every API key read as empty -- which
# surfaced as a 503 on /api/voice that looked like a missing credential rather
# than a missing file. Anchoring removes the whole class of failure.
# override=False so a real environment variable, as set by a host's secret
# manager, always beats a stale local file.
load_dotenv(BACKEND_DIR.parent / ".env", override=False)
load_dotenv(BACKEND_DIR / ".env", override=False)
DATA_DIR = BACKEND_DIR / "data" / "msmarco-xi" / "validation"
ARTIFACT_DIR = BACKEND_DIR / "artifacts"

INDEX_PATH = ARTIFACT_DIR / "vector_index.faiss"
METADATA_PATH = ARTIFACT_DIR / "metadata.pkl"

# Language -> local parquet file. English is carried inside every file as the
# aligned `English_passages` column, so it needs no file of its own.
LANG_FILES: dict[str, str] = {
    "hi": "hinval.parquet",
    "ta": "tamval.parquet",
}
SUPPORTED_LANGS: tuple[str, ...] = ("hi", "ta", "en")

# --- server -----------------------------------------------------------------

CORS_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    if origin.strip()
]

# --- embedding --------------------------------------------------------------

# Quantized ONNX, runs locally on CPU: no network hop, ~10-15ms per query.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# bge-small-en-v1.5 was trained with an instruction prefix on the query side
# only. Passages are embedded bare; omitting this on queries costs real recall.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Small batches on purpose: the build target is a 2-core / low-RAM machine,
# where large batches raise peak memory enough to stall the pass entirely.
EMBED_BATCH_SIZE = 32

# --- FAISS HNSW -------------------------------------------------------------
# Vectors are L2-normalized, so inner product == cosine similarity.

HNSW_M = 32  # neighbours per node; 32 is the accuracy/memory knee for <1M vectors
HNSW_EF_CONSTRUCTION = 200  # build-time depth; higher = better graph, slower build
HNSW_EF_SEARCH = 64  # query-time depth; the main recall-vs-latency dial

DEFAULT_TOP_K = 5

# --- guardrails -------------------------------------------------------------

# Below this cosine score the retrieved context is treated as ungrounded and the
# request is refused without calling the LLM.
#
# Calibrated empirically against the built index (25 real dataset queries vs 15
# deliberately unrelated ones), not taken from a spec sheet. bge-small-en-v1.5
# compresses cosine scores into a narrow high band, so the measured
# distributions were:
#
#   on-topic  : min 0.6536, median 0.8147, max 0.9037
#   off-topic : min 0.5688, median 0.6154, max 0.7764
#
#   threshold   on-topic kept   off-topic leaked
#   0.38               100%               100%   <- inert, never fires
#   0.60               100%                80%
#   0.65               100%                33%   <- chosen: the knee
#   0.75                80%                 7%
#
# 0.65 is the highest threshold that still refuses no genuine query. The bands
# overlap, so no threshold separates them perfectly; raising it trades real
# recall for less leakage. Re-run the calibration if the index size or the
# embedding model changes, since the score distribution moves with both.
SIMILARITY_THRESHOLD = float(os.getenv("EMBEDDING_SIMILARITY_THRESHOLD", "0.65"))

# Longer inputs are rejected outright: a genuine spoken or typed question never
# needs this much room, and unbounded input is both an embedding cost and an
# injection surface.
MAX_QUERY_CHARS = 1000

# Shown verbatim to the user when grounding fails. Never a fabricated answer:
# in a retrieval product a confident wrong answer is worse than a refusal.
UNGROUNDED_MESSAGE = "Context not found"

# --- generation (Groq LPU) --------------------------------------------------

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# llama-3.1-8b-instant is no longer served by Groq (404 on this account), so the
# replacement was picked by measuring the models actually available, from India,
# on TTFT *and* Hindi output quality:
#
#   model                 en TTFT   hi TTFT   Hindi output
#   openai/gpt-oss-20b      659ms     617ms   clean            <- chosen
#   groq/compound-mini      866ms    1194ms   clean
#   qwen/qwen3.6-27b        148ms     243ms   leaks <think> blocks
#   allam-2-7b              158ms     128ms   garbled (Arabic-tuned)
#
# The two fast models fail the requirement that actually matters for an Indic
# RAG product. Re-check availability before a demo; Groq retires models.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Passages sent as context. More context costs prompt-processing time on every
# request, and the top few chunks carry nearly all the answer signal.
MAX_CONTEXT_CHUNKS = 4

# openai/gpt-oss-20b is a reasoning model: it can spend this whole budget on
# internal reasoning (returned in a separate `reasoning` field) before ever
# emitting visible `content`, which surfaces as an empty response with no
# retry possible -- the tokens are already spent. Groq's own default for
# max_completion_tokens is 1024, with their docs noting even that "may be too
# low for complex reasoning"; this was set to half that. Raised to match
# their default, paired with reasoning_effort="low" in the harness to reduce
# how much budget reasoning itself consumes.
MAX_OUTPUT_TOKENS = 1024

# --- speech to text (Sarvam) ------------------------------------------------

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_TRANSCRIBE_URL = "https://api.sarvam.ai/speech-to-text"

# saaras returns ENGLISH text plus the detected source language. That is exactly
# what the English vector space needs, so a Hindi or Tamil question becomes
# directly searchable without a separate translation hop.
# Measured: saaras:v3 506ms and reports language_code; saaras:v2.5 434ms but
# returned language_code=None, so v3 earns its slightly higher cost.
SARVAM_TRANSLATE_MODEL = os.getenv("SARVAM_TRANSLATE_MODEL", "saaras:v3")

# saarika returns the NATIVE script, used only to show users their own words.
# saarika:v2 and saaras:v2 are both deprecated upstream and return 400.
SARVAM_TRANSCRIBE_MODEL = os.getenv("SARVAM_TRANSCRIBE_MODEL", "saarika:v2.5")

# Sarvam reports e.g. "hi-IN"; the pipeline speaks "hi". Anything outside the
# supported set is answered in English rather than guessed at.
SARVAM_LANG_MAP: dict[str, str] = {"hi-IN": "hi", "ta-IN": "ta", "en-IN": "en"}

# ~30 seconds of 16kHz mono PCM. A spoken question never needs more, and
# unbounded uploads are both a cost and an abuse surface.
MAX_AUDIO_BYTES = 2_000_000
STT_TIMEOUT_S = float(os.getenv("STT_TIMEOUT_S", "20.0"))

# --- speech-to-text fallback (Groq Whisper) ---------------------------------
#
# Sarvam is the primary: it is purpose-built for Indic speech and returns the
# detected language. Whisper on Groq is the standby, used when Sarvam has no
# key, rejects the key, or fails. Two providers on different vendors means one
# outage does not take voice input down during a demo.
#
# Translations run on whisper-large-v3 because the turbo variant is
# transcription-only; transcription itself uses turbo, which is faster.
GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_TRANSLATE_URL = "https://api.groq.com/openai/v1/audio/translations"
GROQ_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
GROQ_WHISPER_TRANSLATE_MODEL = os.getenv("GROQ_WHISPER_TRANSLATE_MODEL", "whisper-large-v3")

# --- text translation -------------------------------------------------------

# A TYPED Hindi or Tamil question still has to reach an English vector space.
# Voice already arrives translated via saaras, so this hop only applies to typed
# Indic input; a Latin-script query skips it entirely and pays nothing.
# Measured: mayura:v1 243-271ms, sarvam-translate:v1 246-392ms.
SARVAM_TEXT_TRANSLATE_URL = "https://api.sarvam.ai/translate"
SARVAM_TEXT_TRANSLATE_MODEL = os.getenv("SARVAM_TEXT_TRANSLATE_MODEL", "mayura:v1")
LANG_TO_SARVAM_CODE: dict[str, str] = {"hi": "hi-IN", "ta": "ta-IN", "en": "en-IN"}

# --- latency budget ---------------------------------------------------------

# Target time-to-first-token. This is a BUDGET, not a kill switch: responses are
# tagged `within_budget` and the real figure is reported, rather than aborting a
# working answer for being 20ms late. Aborting would turn a slow success into a
# user-visible failure and make the reported latency a lie by omission.
TTFT_BUDGET_MS = int(os.getenv("LLM_TIMEOUT_MS", "150"))

# The actual kill switch, well above the budget, so a genuinely hung upstream
# cannot pin a request open forever.
REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", "10.0"))
CONNECT_TIMEOUT_S = 3.0

# Circuit breaker: after this many consecutive upstream failures, fail fast for
# the cooldown instead of making every user wait for the same timeout.
CIRCUIT_FAILURE_THRESHOLD = 3
CIRCUIT_RECOVERY_S = 15.0

# Retries only help on connection-level faults; a timeout has already spent the
# budget, so retrying it just doubles the wait.
MAX_RETRIES = 1
