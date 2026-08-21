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

**Voice out as well as in.** An answer can be spoken back through Sarvam
bulbul, in the language it was answered in. A question asked by voice is
answered aloud by default; a typed one is not, because reading the screen is
not asking for sound and synthesis costs a round trip. Both are a button next
to the answer. Bulbul rather than the browser's `speechSynthesis` because
Hindi and Tamil voices are absent on most Windows installs, so the two
languages this project exists for would fall back to an English voice reading
Devanagari.

**The model backend is a switch, not a rebuild.** Groq is the default and what
a deployment uses. If Ollama is running locally with the model pulled, the
header offers it and generation moves to your own machine: measured at 82ms to
first token against Groq's 438ms, because there is no network hop. The switch
appears only when the backend can actually answer -- installed, running and
model present are three different things -- and says how to fix it when it
cannot.

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
781MB of artifacts, **4.6s cold start** (unpickling 445MB of chunk payloads
plus the ONNX warmup) and **~1.06GB resident**. Those are the figures that
decide whether a host can run this, not the per-query numbers.

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

### Where the 200ms actually goes: Groq vs a local model

The sub-200ms target is missed because of the network, not the pipeline. To
measure that rather than assert it, the harness can be pointed at a model
running on this machine — `LLM_PROVIDER=ollama`, same prompts, same retrieved
context, same streaming parse. Groq stays the default and the deployment
target; this is a measurement, not a migration.

| Generation stage only | Groq `gpt-oss-20b` | Local `llama3.2:3b` |
| --- | ---: | ---: |
| **TTFT P50** | 438ms | **82ms** |
| TTFT P100 | 537ms | **96ms** |
| Total P50 | 534ms | 497ms |
| Total P100 | **715ms** | 808ms |

**Time to first token is 5.3x faster locally; total time is a wash.** That
split is the whole story: TTFT is dominated by the round trip to Groq, which
localhost does not pay, while Groq's LPU then decodes fast enough to claw the
difference back over a full answer.

End-to-end, the local model is the only configuration measured here that meets
the target — and only for first token:

| End-to-end to first token | P50 | P70 | P90 | P100 |
| --- | ---: | ---: | ---: | ---: |
| guardrail + embed + FAISS + TTFT | **177.6ms** | **186.4ms** | 214.5ms | 219.0ms |

**Sub-200ms is met at P50 and P70, and missed from P90 up.** Three things stop
this from being the answer to the requirement rather than a data point:

- It is **time to first readable token, not the finished answer** — full
  completion is 369ms P50 locally. "Through to final output" is still not met
  by any configuration measured.
- It **needs a local GPU**. The deployment target's free tier has none, so
  running this configuration there would be slower than Groq, not faster.
- It is a **3B model against a 20B one**. Answer quality held up on this
  corpus — 0 refusals and 0 errors across 8 identical prompts, with comparable
  groundedness — but that is a narrow test, not a quality claim.

Reproduce with `python compare_providers.py` (needs Ollama running and
`ollama pull llama3.2:3b`).

Reproduce with `python benchmark.py --queries 100 --generation-samples 20`
(close the browser first — the WebGL canvas competes for CPU with the process
being measured).

---

## Chunking strategies

Four strategies are implemented and **scored against each other**, using the
corpus's own `is_selected` flag as ground truth rather than any judgement of
our own.

Scored over **100 queries across 997 passages** (the earlier published run used
25 queries, small enough that a 0.1 gap was three queries and inside the
noise).

| Strategy | Chunks | Vector MB | R@1 | R@3 | R@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed size | 1,091 | 1.68 | 0.360 | 0.770 | **0.930** | 0.579 |
| Fixed + overlap | 1,091 | 1.68 | 0.360 | 0.780 | 0.920 | 0.580 |
| **Semantic** | 2,100 | 3.23 | **0.470** | 0.730 | 0.880 | **0.626** |
| Hierarchical | 2,100 | 3.23 | 0.420 | 0.760 | 0.800 | 0.587 |

Three findings worth stating plainly:

- **Semantic leads where it matters for RAG.** It takes R@1 (0.470 vs 0.360)
  and MRR@5, and top-rank precision is what decides which passage actually
  reaches the model. It pays for that with double the chunks and double the
  vector memory — a win, but not a free one.
- **Fixed wins R@5** (0.930) because larger chunks cover more ground once five
  results are allowed. Which strategy is "best" depends entirely on how many
  results you feed the model; at `MAX_CONTEXT_CHUNKS = 4` the top-rank metrics
  are the ones that count.
- **Hierarchical still scores below plain semantic** despite injecting parent
  context — 0.420 vs 0.470 R@1, on identical chunk counts. At 25 queries this
  was dismissible as noise. At 100 it is five queries out of a hundred and
  reproduces the earlier direction, so the parent-context injection is not
  paying for itself here.

The two fixed strategies are now *nearly* identical rather than exactly so.
Passages measure p50 294 characters against a 480-character window, so most
stay unsplit and overlap never engages; only the minority long enough to split
differ at all (R@3 0.780 vs 0.770, R@5 0.920 vs 0.930). Window size is a
parameter for exactly this reason.

Reproduce with `python -m app.chunk_eval --queries 100` — the results cache to
`artifacts/chunk_comparison.json`, which is what the **Chunking Explorer** tab
reads. That file is a build output and is not committed, so the tab shows an
empty comparison until the command has been run once.

---

## Guardrails

| Stage | Catches | Cost |
| --- | --- | ---: |
| Pre-retrieval | Prompt injection, role hijack, prompt extraction, delimiter injection | 0.04ms |
| Pre-retrieval | Harmful instructions, PII (Luhn-validated cards, Aadhaar, phone, email) | 0.04ms |
| Pre-retrieval | Greetings and small talk, answered directly | 0.1ms |
| Post-retrieval | Cosine similarity below threshold | 0.02ms |
| Post-generation | The model itself declining for lack of usable context | — |

**"hi" is not a question, and treating it as one looked like a bug.** Sent
down the retrieval path it embeds, matches a passage about the Japanese kana
は at 0.7036 — above the threshold — and reaches the model, which spends 1.1
seconds and real tokens correctly concluding the passage does not answer it.
The user's first input gets a red refusal. It was behaving exactly as
designed and the design was wrong.

Greetings, thanks, farewells and "what can you do" are now answered before any
embedding, in the user's language, in **0.1ms and zero tokens**. The patterns
are anchored to the whole string, so *"hello, what is inflation"* is still a
question and still goes to retrieval — only a bare greeting short-circuits.

**The threshold was calibrated twice, not assumed.** The original 0.38
admitted 100% of off-topic queries — `bge-small-en-v1.5` compresses cosine
scores into a narrow high band, so a threshold set for a wider-spread model
never fires. Recalibrating against the 5,289-vector index gave 0.65.

**Growing the index 37x invalidated that.** With 194,904 chunks almost any
query finds *some* neighbour above a low bar, and 0.65 went most of the way
inert again — it admitted 80% of deliberately unanswerable queries. So it was
calibrated a second time, against two query sets rather than one:

```
on-topic, corpus-verbatim (n=60)   min 0.6180   median 0.8223   max 0.9441
on-topic, natural phrasing (n=15)  min 0.6924   median 0.8221   max 0.8901
off-topic, unanswerable   (n=15)   min 0.6329   median 0.6720   max 0.7856

threshold   natural kept   off-topic leaked
0.65               100%              80%   <- previous, largely inert
0.68               100%              47%   <- chosen
0.70                93%              40%
0.80                58%               0%
```

Corpus-verbatim queries alone are misleading: they are the exact strings the
passages were written for, so they score high and make any threshold look
safe. Natural phrasing is what decides the value. 0.70 was measured first and
rejected — it buys 7 points of leakage at the cost of falsely refusing *"Who
is Obama?"* at 0.6924, a question this corpus answers well.

The bands still overlap, so roughly half of unanswerable queries reach the
model. **That is what the post-generation check is for, and it demonstrably
works:** *"how do purple elephants photosynthesize underwater"* clears
retrieval at 0.7241 and comes back `Context not found` from the model itself,
reported as `model_refused: true` rather than dressed up as a grounded answer.
Two independent judges, and the cheap one is a pre-filter rather than the only
line of defence.

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
- **The chunking comparison is a build output and is not committed.** The
  Chunking Explorer's comparison view is empty on a fresh clone until
  `python -m app.chunk_eval --queries 100` has been run once, which takes about
  ten minutes of CPU embedding.
- **The index covers 10,000 source rows** (194,904 vectors) of the 97,941
  available. Coverage is broad but not complete: a question whose supporting
  passage falls in the remaining rows is refused as ungrounded, which looks
  identical to a failure from the outside. `python -m app.indexer --rows N`
  rebuilds it larger, at roughly 5.4 hours per 10,000 rows on a 10-core CPU.
- **The artifacts are 781MB and cannot live in the repository.** GitHub hard
  rejects files over 100MB, so they ship as Release assets that the setup
  scripts fetch. Loading them costs ~1.06GB of RAM, which rules out the
  512MB-and-under free tiers on several hosts.
- **Setup depends on this repository staying public.** Release assets on a
  private repository return 404 to an unauthenticated download, so making it
  private again would break `setup.ps1` and `setup.sh` for everyone without
  any error message pointing at the cause.
- **No authentication or rate limiting.** The deployment is anonymous.
