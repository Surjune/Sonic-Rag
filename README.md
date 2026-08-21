# Sonic-RAG

**Voice-enabled Indic retrieval-augmented generation.**
Hacker House Goa 2026 · Task 2 · Team **Lightning Logics**

Ask a question by voice or text in **English, Hindi or Tamil** and get an answer
grounded in retrieved passages — or an honest refusal when nothing relevant was
found.

---

## Quick start

The prebuilt FAISS index is **downloaded automatically by the setup script**
from a GitHub Release, so there is **no dataset download and no index
rebuild** — the embedding pass takes 5.4 hours and nobody evaluating this
project should have to run it. Setup is dependencies, a ~781MB artifact
download, and two API keys.

```bash
git clone https://github.com/Surjune/Sonic-Rag.git
cd Sonic-Rag
```

**Windows**
```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

**macOS / Linux**
```bash
bash setup.sh
```

Then put your keys in `.env` (both have free tiers):

| Key | Used for | Get one at |
| --- | --- | --- |
| `GROQ_API_KEY` | Answer generation, and the Whisper speech fallback | console.groq.com |
| `SARVAM_API_KEY` | Indic speech-to-text and text translation | dashboard.sarvam.ai |

Run it, in two terminals:

```powershell
# 1 — backend
cd backend
.\.venv\Scripts\Activate.ps1      # macOS/Linux: source .venv/bin/activate
uvicorn app.main:app --port 8000

# 2 — frontend
cd frontend
npm run dev
```

Open **http://localhost:5173**. API docs are at http://localhost:8000/docs.

> Port 8000 already taken?
> `Get-NetTCPConnection -LocalPort 8000 -State Listen | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }`

---

## Requirements

- **Python 3.11+**, **Node 18+**
- ~2GB disk for dependencies (ONNX runtime and Three.js dominate)
- No GPU required — embeddings run on CPU via quantized ONNX

---

## How it works

```
voice ──▶ Sarvam saaras ──┐
                          ├──▶ input guardrail ──▶ embed ──▶ FAISS ──▶ grounding ──▶ Groq ──▶ answer
text ───▶ translate ──────┘         │                                      │
                                (blocks)                              (refuses)
```

Both guardrails short-circuit, so a blocked or ungrounded request never reaches
the model and spends no tokens.

**One English vector space.** MSMARCO-XI is a parallel corpus, so English is
embedded once and the Hindi and Tamil passages ride along as display payloads.
`bge-small-en-v1.5` has an English wordpiece vocabulary — Devanagari and Tamil
would tokenize to `[UNK]` and embed meaninglessly. Users still ask and read in
their own language; only the vector space is English.

---

## Measured latency

100 real corpus queries, sequential, against the **194,904-vector index**, on
a **10-core i5-13450HX / 16GB** — a much stronger machine than the original
2-core i3 floor, so these show what the pipeline does when hardware is not the
bottleneck.

| Stage | P50 | P70 | P95 | P100 |
| --- | ---: | ---: | ---: | ---: |
| Input guardrail | 0.03ms | 0.03ms | 0.04ms | 0.23ms |
| Embedding | 53.30ms | 54.69ms | 59.61ms | 62.14ms |
| **FAISS search** | **0.38ms** | 0.43ms | 0.54ms | 1.04ms |
| Grounding check | 0.01ms | 0.01ms | 0.01ms | 0.02ms |
| **Retrieval total** | **53.80ms** | **55.23ms** | **60.11ms** | 62.73ms |

- **Injection blocked: 0.02ms P50** — refused before any embedding or network call
- **Ungrounded refusal: 54.97ms P50** — no model call, no tokens spent
- **Typed Hindi retrieval: 346.49ms P50** (adds the Sarvam text-translation hop, 293.74ms P50 of that)
- **Typed Tamil retrieval: 318.97ms P50** (translation hop 258.98ms P50)

**Index size barely moves retrieval.** Growing the index 37x, from 5,289 to
194,904 vectors, moved FAISS search from 0.22ms to 0.38ms P50 and left total
retrieval flat at ~54ms. Embedding is a fixed-cost forward pass that does not
care how large the index is, and HNSW search is logarithmic in it, so neither
scales the way intuition suggests. What grows instead is startup and memory:
781MB of artifacts and ~1.06GB resident.

**Full pipeline, including generation** (n=18 of 20; the other two hit a Groq
free-tier 429 mid-run, not a pipeline fault):

| Stage | P50 | P70 | P90 | P95 | P100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Embedding | 55.51ms | 58.52ms | 60.12ms | 63.54ms | 63.54ms |
| FAISS search | 0.42ms | 0.44ms | 0.57ms | 0.84ms | 0.84ms |
| **Groq TTFT** | **468.58ms** | 493.05ms | 611.67ms | 640.48ms | 640.48ms |
| Groq total generation | 528.46ms | 579.54ms | 721.66ms | 761.41ms | 761.41ms |
| **Full pipeline total** | **578.13ms** | **634.86ms** | **777.11ms** | **814.55ms** | 814.55ms |

**Honest reading:** retrieval alone comfortably clears sub-200ms, even at
P100. The full pipeline does not, at any percentile, once a network-bound
model call is involved — and no tech-stack change fixes that from this side:
Groq's own TTFT is the round trip to their infrastructure, not local compute.

`openai/gpt-oss-20b` is a reasoning model, and it previously spent its entire
`MAX_OUTPUT_TOKENS` budget on internal reasoning before emitting any visible
content on some queries, returning nothing at all (3 of 20 attempts in an
earlier run). Fixed by capping `reasoning_effort` to `"low"` and raising
`MAX_OUTPUT_TOKENS` from 512 to 1024, which also cut P50 by ~160ms. The two
failures in the run above are a different thing entirely — quota exhaustion
from repeated benchmarking, where the harness rotated to the backup key and
then opened the circuit, exactly as designed.

Reproduce with `python benchmark.py --queries 100 --generation-samples 20`
(close the browser first — the WebGL canvas competes for CPU with the process
being measured).

---

## Chunking strategies

Four strategies are implemented and **scored against each other**, using the
corpus's own `is_selected` flag as ground truth rather than any judgement of
our own.

| Strategy | Chunks | R@1 | R@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: |
| Fixed size | 276 | 0.440 | **0.920** | 0.621 |
| Fixed + overlap | 276 | 0.400 | 0.880 | 0.593 |
| **Semantic** | 524 | **0.600** | 0.880 | **0.701** |
| Hierarchical | 524 | 0.480 | 0.800 | 0.610 |

Two findings worth stating plainly:

- **Semantic leads where it matters for RAG** — top-rank precision decides what
  actually reaches the model. Fixed wins R@5 because larger chunks cover more
  ground once five results are allowed.
- **Hierarchical scored worse than plain semantic**, despite injecting parent
  context. At 25 queries a 0.12 gap is three queries and inside the noise, so it
  is flagged for a larger run rather than acted on.

The two fixed strategies produced identical output because passages measure
p50 294 and p90 478 characters against a 480-character window, leaving 90%
unsplit so overlap never engages. Window size is a parameter for this reason.

Reproduce with `python -m app.chunk_eval --queries 25`, or explore boundaries
live in the **Chunking Explorer** tab.

---

## Guardrails

| Stage | Catches | Cost |
| --- | --- | ---: |
| Pre-retrieval | Prompt injection, role hijack, prompt extraction, delimiter injection | 0.04ms |
| Pre-retrieval | Harmful instructions, PII (Luhn-validated cards, Aadhaar, phone, email) | 0.04ms |
| Post-retrieval | Cosine similarity below threshold | 0.02ms |
| Post-generation | The model itself declining for lack of usable context | — |

**The threshold was calibrated, not assumed.** The original 0.38 admitted
**100%** of deliberately off-topic queries — a chocolate cake recipe scored
0.6157 and would have been passed to the model as grounded context.
`bge-small-en-v1.5` compresses cosine scores into a narrow high band, so a
threshold set for a wider-spread model never fires. Measured distributions:

```
on-topic   min 0.6536   median 0.8147   max 0.9037
off-topic  min 0.5688   median 0.6154   max 0.7764

threshold   on-topic kept   off-topic leaked
0.38               100%               100%   <- inert
0.65               100%                33%   <- chosen
0.75                80%                 7%
```

0.65 is the highest value that still refuses no genuine query. The bands
overlap, so no threshold separates them perfectly and roughly a third of
off-topic queries still get through — claiming perfect grounding would be false.

**That calibration was done against the 250-row index, and the corpus has
since grown 37x.** Re-testing on the 194,904-vector index, every query in the
old off-topic set now clears the threshold: a chocolate cake recipe scores
0.7367 where it once scored 0.6157. Inspecting what comes back shows this is
correct rather than a regression — the retrieved passages really are a cake
recipe, real German Shepherd training advice, real guitar-tuning instructions.
MSMARCO is a broad web corpus, so at 10,000 rows those topics are genuinely
covered and the queries are no longer off-topic for it. What this invalidates
is the *test set*, not the threshold: `benchmark.py`'s `OFF_TOPIC_QUERIES`
list was written for a corpus where those subjects were absent, and it needs
replacing with queries that are actually outside the larger index before the
refusal path can be honestly measured again.

Unsafe-content rules match **actionable instructions, not topics**: *"how do I
build a bomb"* is blocked, *"how do explosives work in mining"* is not.
Over-blocking ordinary questions is a silent failure, so benign queries are
tested in their own right.

---

## Resilience

- **Two speech providers.** Sarvam leads; Groq Whisper takes over on a missing
  key, 401/403, timeout or 5xx. Whisper is faster (300ms vs 896ms) but hears
  Hindi as Urdu and returns Arabic script, so it is a standby, not a peer — the
  UI language selection is passed as a hint to correct it.
- **Key rotation.** Free-tier quota is per key, so `*_API_KEY_BACKUP` rotates in
  on 401/403/429 before any vendor failover. A 500 does not rotate; the upstream
  is unwell and a different credential will not persuade it.
- **Circuit breaker.** After repeated upstream failures the harness fails fast
  rather than making every user wait out the same timeout.
- **No fabrication.** Missing credentials and unreachable upstreams produce typed
  errors. A confident wrong answer is worse than an honest failure.

---

## Testing

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
pytest tests\ -q          # 258 tests, no network calls
```

Every external client is mocked via `httpx.MockTransport`.

---

## Layout

```
backend/
  app/
    main.py          FastAPI routes, SSE streaming, per-stage telemetry
    retrieval.py     FAISS + embedding singleton
    chunkers.py      four strategies behind one interface
    chunk_eval.py    Recall@K / MRR scoring across strategies
    guardrails.py    injection, unsafe content, PII, grounding
    harness.py       Groq client, tools, retries, circuit breaker
    stt_service.py   Sarvam primary, Whisper fallback
    credentials.py   key rotation
  artifacts/         prebuilt FAISS index (committed)
  benchmark.py       P50/P70/P100 profiler
frontend/
  src/
    components/Orb.tsx   audio-reactive GLSL visualizer
    tabs/                playground, analytics, chunking, guardrails
```

---

## Known limitations

- **Sub-200ms is not met end-to-end** with generation, only for retrieval —
  542ms P50 / 1739ms P100 full pipeline vs. 53ms P50 / 66ms P100 retrieval
  alone. See the latency section.
- **Hindi and Tamil injection patterns are a narrow starter set** and want a
  native-speaker review before being relied on.
- **The chunking comparison ran on 25 queries.** Differences of ~0.1 are within
  noise; a larger run is needed before acting on the hierarchical result.
- **The index covers 10,000 source rows** (194,904 vectors) of the 97,941
  available. Coverage is broad but not complete: a question whose supporting
  passage falls in the remaining rows is refused as ungrounded, which looks
  identical to a failure from the outside. `python -m app.indexer --rows N`
  rebuilds it larger, at roughly 5.4 hours per 10,000 rows on a 10-core CPU.
- **The artifacts are 781MB and cannot live in the repository.** GitHub hard
  rejects files over 100MB, so they ship as Release assets that the setup
  scripts fetch. Loading them costs ~1.06GB of RAM, which rules out the
  512MB-and-under free tiers on several hosts.
- **No authentication or rate limiting.** The deployment is anonymous.
