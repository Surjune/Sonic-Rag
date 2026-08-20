# Sonic-RAG

**Voice-enabled Indic retrieval-augmented generation.**
Hacker House Goa 2026 · Task 2 · Team **Lightning Logics**

Ask a question by voice or text in **English, Hindi or Tamil** and get an answer
grounded in retrieved passages — or an honest refusal when nothing relevant was
found.

---

## Quick start

The FAISS index ships with this repository, so there is **no dataset download
and no index rebuild**. Setup is dependencies plus two API keys.

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

100 real corpus queries, sequential, on a **loaded 2-core i3** — a weak machine,
so treat these as a floor rather than a best case.

| Stage | P50 | P70 | P95 | P100 |
| --- | ---: | ---: | ---: | ---: |
| Input guardrail | 0.04ms | 0.04ms | 0.05ms | 0.06ms |
| Embedding | 116.29ms | 124.43ms | 150.94ms | 208.48ms |
| **FAISS search** | **0.41ms** | 0.44ms | 0.54ms | 0.78ms |
| Grounding check | 0.02ms | 0.02ms | 0.03ms | 0.04ms |
| **Retrieval total** | **116.96ms** | **124.94ms** | **151.46ms** | 209.31ms |

- **Injection blocked: 0.05ms** — refused before any embedding or network call
- **Ungrounded refusal: 104ms** — no model call, no tokens spent
- **Full generation: 917ms P50**, TTFT 659ms

**Honest reading:** the retrieval pipeline meets sub-200ms through P95. It does
**not** meet it once a network-bound model call is involved, and no free-tier
configuration changes that — TTFT is dominated by the round trip, and variance
between repeated calls to the same model (814–1337ms) exceeds the difference
between models.

Reproduce with `python benchmark.py --queries 100` (close the browser first —
the WebGL canvas competes for CPU with the process being measured).

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

- **Sub-200ms is not met end-to-end** with generation, only for retrieval. See
  the latency section.
- **Hindi and Tamil injection patterns are a narrow starter set** and want a
  native-speaker review before being relied on.
- **The chunking comparison ran on 25 queries.** Differences of ~0.1 are within
  noise; a larger run is needed before acting on the hierarchical result.
- **The index covers 250 source rows** (5,289 vectors), sized for a demo rather
  than for coverage. `python -m app.indexer --rows N` rebuilds it larger.
- **No authentication or rate limiting.** The deployment is anonymous.
