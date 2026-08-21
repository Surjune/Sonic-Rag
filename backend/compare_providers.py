"""Measure Groq against a local Ollama model on the generation stage alone.

Retrieval is identical for both -- same index, same embedding, same FAISS
search -- so including it would add the same constant to each column and make
the difference look smaller than it is. This drives the harness directly with
contexts retrieved once up front, so what is measured is the thing that
actually differs: where the model runs.

Reported per provider:

    ttft     time to first token, which is what a reader experiences as "fast"
    total    time to the last token, a throughput number rather than a latency
    tokens   streamed chunk count, as a sanity check that answers are comparable
    refused  the model declining for lack of usable context

The point is not to declare a winner. A 3B model on a laptop GPU and a 20B
model on Groq's LPUs are not the same product, and a latency number that comes
with a quality collapse has not won anything. Both columns are printed so the
tradeoff stays visible.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from dataclasses import dataclass, field

from app.config import DEFAULT_TOP_K, GROQ_MODEL, OLLAMA_MODEL
from app.exceptions import SonicRagError
from app.harness import GenerationRequest, GroqHarness
from app.retrieval import engine

QUERIES: list[str] = [
    "what is a corporation",
    "what causes diabetes",
    "how does photosynthesis work",
    "what is inflation",
    "how do vaccines work",
    "what does dna do",
    "how are earthquakes measured",
    "what is a mortgage",
]


@dataclass
class Sample:
    ttft_ms: float
    total_ms: float
    tokens: int
    refused: bool
    text: str = ""


@dataclass
class Column:
    provider: str
    model: str
    samples: list[Sample] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def stat(self, attr: str, fraction: float) -> float:
        values = sorted(getattr(s, attr) for s in self.samples)
        if not values:
            return 0.0
        if fraction >= 1.0:
            return values[-1]
        rank = max(1, min(len(values), int(-(-fraction * len(values) // 1))))
        return values[rank - 1]

    def mean(self, attr: str) -> float:
        values = [getattr(s, attr) for s in self.samples]
        return statistics.fmean(values) if values else 0.0


async def measure(harness: GroqHarness, prompts: list[GenerationRequest]) -> Column:
    column = Column(provider=harness.provider, model=harness.model)
    # Warm the backend so model load or TLS setup is not charged to query one.
    await harness.warmup()
    try:
        await harness.generate(prompts[0])
    except SonicRagError:
        pass

    for request in prompts:
        try:
            result = await harness.generate(request)
        except SonicRagError as error:
            column.errors.append(f"{error.__class__.__name__}: {error}")
            continue
        column.samples.append(
            Sample(
                ttft_ms=result.ttft_ms,
                total_ms=result.total_ms,
                tokens=result.tokens,
                refused=result.model_refused,
                text=result.text,
            )
        )
    return column


def report(columns: list[Column]) -> str:
    lines: list[str] = ["", "## Generation stage: Groq vs local Ollama", ""]
    lines.append("| metric | " + " | ".join(f"{c.provider} ({c.model})" for c in columns) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in columns) + " |")

    rows = [
        ("TTFT P50", lambda c: f"{c.stat('ttft_ms', 0.50):.0f}ms"),
        ("TTFT P90", lambda c: f"{c.stat('ttft_ms', 0.90):.0f}ms"),
        ("TTFT P100", lambda c: f"{c.stat('ttft_ms', 1.00):.0f}ms"),
        ("Total P50", lambda c: f"{c.stat('total_ms', 0.50):.0f}ms"),
        ("Total P100", lambda c: f"{c.stat('total_ms', 1.00):.0f}ms"),
        ("mean tokens", lambda c: f"{c.mean('tokens'):.0f}"),
        ("refusals", lambda c: f"{sum(1 for s in c.samples if s.refused)}/{len(c.samples)}"),
        ("errors", lambda c: str(len(c.errors))),
    ]
    for name, fn in rows:
        lines.append(f"| {name} | " + " | ".join(fn(c) for c in columns) + " |")

    lines.append("")
    lines.append("### Sample answers, same question and context")
    for column in columns:
        answer = column.samples[0].text if column.samples else "(no answer)"
        lines.append(f"\n**{column.provider} / {column.model}**\n\n> {answer[:400]}")
    for column in columns:
        for problem in column.errors[:3]:
            lines.append(f"\n`{column.provider}` error: {problem}")
    return "\n".join(lines)


async def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Compare Groq against local Ollama.")
    parser.add_argument("--queries", type=int, default=len(QUERIES))
    parser.add_argument("--ollama-model", default=OLLAMA_MODEL)
    parser.add_argument("--groq-model", default=GROQ_MODEL)
    args = parser.parse_args()

    print("loading index...")
    engine.load()

    # Retrieve once. Both providers must answer from identical context or the
    # comparison measures retrieval variance instead of the model.
    prompts: list[GenerationRequest] = []
    for query in QUERIES[: args.queries]:
        vector = engine.embed_query(query)
        hits = engine.search(vector, DEFAULT_TOP_K)
        prompts.append(
            GenerationRequest(
                query=query, contexts=engine.build_contexts(hits), language="en"
            )
        )
    print(f"prepared {len(prompts)} prompts from identical retrieved context\n")

    columns: list[Column] = []
    for provider, model in (("groq", args.groq_model), ("ollama", args.ollama_model)):
        print(f"measuring {provider} ({model})...")
        harness = GroqHarness(provider=provider, model=model)
        try:
            columns.append(await measure(harness, prompts))
        finally:
            await harness.aclose()

    print(report(columns))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
