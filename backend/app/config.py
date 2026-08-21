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
# Recalibrated for the 194,904-vector index. The previous 0.65 was measured
# against a 5,289-vector index, and a threshold does not survive a 37x corpus
# change: with far more chunks almost any query finds *some* neighbour above a
# low bar, so the old value had gone most of the way inert -- it admitted 80%
# of deliberately unanswerable queries.
#
# Calibrated against two query sets, not one. Corpus-verbatim queries alone are
# misleading: they are the exact strings the passages were written for, so they
# score high and make any threshold look safe. The set that decides the value is
# natural phrasing -- how somebody actually asks -- which scores lower for the
# same answerable question.
#
#   on-topic, corpus-verbatim (n=60) : min 0.6180, median 0.8223, max 0.9441
#   on-topic, natural phrasing (n=15): min 0.6924, median 0.8221, max 0.8901
#   off-topic, unanswerable (n=15)   : min 0.6329, median 0.6720, max 0.7856
#
#   threshold   natural kept   off-topic leaked
#   0.65              100%              80%   <- previous, largely inert
#   0.68              100%              47%   <- chosen: the knee
#   0.70               93%              40%
#   0.72               87%              33%
#   0.80               58%               0%
#
# 0.68 is the highest threshold that still refuses no genuine query, which is
# the same rule the original 0.65 was chosen by. 0.70 was measured first and
# rejected: it buys 7 points of leakage for a false refusal of "Who is Obama?"
# at 0.6924, a question this corpus answers perfectly well. A false refusal on
# an answerable question is worse than spending tokens on junk the model will
# refuse anyway.
#
# The bands still overlap (off-topic reaches 0.7856, natural on-topic starts at
# 0.6924), so no threshold separates them cleanly and roughly half of
# unanswerable queries still reach the model. That is what the post-generation
# check is for, and it is verified rather than hoped for: nonsense that clears
# this threshold -- "how do purple elephants photosynthesize underwater" at
# 0.7241 -- comes back "Context not found" from the model itself. Re-run this
# calibration whenever the index size or the embedding model changes.
SIMILARITY_THRESHOLD = float(os.getenv("EMBEDDING_SIMILARITY_THRESHOLD", "0.68"))

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

# --- local LLM (Ollama), opt-in ------------------------------------------------
#
# Groq remains the default and the deployment target: it is network-bound but
# runs on LPU hardware no laptop matches, and it needs no local GPU. This block
# exists to answer a specific question -- what does removing the network hop
# actually buy? -- by pointing the same harness at a model running on this
# machine. Ollama serves an OpenAI-compatible /v1/chat/completions, so the same
# request shape works against both and the comparison is apples to apples.
#
# Set LLM_PROVIDER=ollama to switch. Nothing about the Groq path changes.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/v1/chat/completions")
# 3B class, quantized: fits an 8GB laptop GPU with room for the KV cache, and
# is the honest comparison point for a free-tier local setup. A larger model
# would win on quality and lose the latency argument this is meant to test.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

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

# --- text to speech (Sarvam bulbul) -------------------------------------------
#
# The counterpart to speech input: a question asked by voice should be
# answerable by voice. Bulbul rather than the browser's speechSynthesis because
# Hindi and Tamil voices are absent on most Windows installs, so the two
# languages this project exists for would fall back to an English voice reading
# Devanagari -- worse than no audio.
#
# Synthesis is per-request and opt-in. It costs a round trip and Sarvam quota,
# and a user reading the answer on screen should not pay for audio nobody plays.
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v2")
SARVAM_TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "anushka")

# bulbul:v2 rejects anything past 1500 characters outright, which would turn a
# long answer into no audio at all. Answers are capped at two or three
# sentences by the system prompt, so this is headroom rather than a limit that
# is expected to bite.
SARVAM_TTS_MAX_CHARS = int(os.getenv("SARVAM_TTS_MAX_CHARS", "1500"))

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
#
# Was 150ms, which no Groq call has ever met -- measured TTFT is 438ms P50 and
# 640ms P90 -- so `within_budget` was false on every single response and the
# interface showed an "over budget" warning permanently. A warning that always
# fires carries no information and trains people to ignore the one time it
# matters.
#
# 700ms is set from the measured distribution rather than from ambition: it
# sits just above Groq's P90, so an ordinary response is silently fine and the
# badge appears when a call is genuinely slower than this backend's normal
# worst case. The aspirational sub-200ms figure has a place, but it is the
# retrieval budget, and retrieval already meets it at 53ms P50 without needing
# a flag to say so.
TTFT_BUDGET_MS = int(os.getenv("LLM_TIMEOUT_MS", "700"))

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
