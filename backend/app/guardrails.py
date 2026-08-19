"""Two-stage guardrails: input filtering before retrieval, grounding after it.

    query -> check_input() -----(block)---> refusal, no embedding, no LLM
               |
             (allow)
               v
           embed + FAISS search
               v
          check_grounding() --(block)---> "Context not found", no LLM
               |
             (allow)
               v
              Groq

Both stages are pure CPU work on short strings and run in microseconds, so a
refusal costs a rounding error rather than a model call. That matters twice
over: it keeps the latency budget intact, and a blocked request never spends a
token against a free-tier quota.

The grounding check exists because a retrieval system that answers from weak
context produces confident, fluent, wrong answers. Refusing is the correct
behavior when nothing relevant was found.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from app.config import MAX_QUERY_CHARS, SIMILARITY_THRESHOLD, UNGROUNDED_MESSAGE


class BlockReason(str, Enum):
    """Machine-readable codes, surfaced in the audit log UI."""

    EMPTY_QUERY = "EMPTY_QUERY"
    QUERY_TOO_LONG = "QUERY_TOO_LONG"
    INSTRUCTION_OVERRIDE = "INSTRUCTION_OVERRIDE"
    ROLE_HIJACK = "ROLE_HIJACK"
    PROMPT_EXTRACTION = "PROMPT_EXTRACTION"
    DELIMITER_INJECTION = "DELIMITER_INJECTION"
    UNGROUNDED = "UNGROUNDED"


# Characters with no visual width, commonly inserted mid-word to break naive
# pattern matching ("ig​nore previous instructions"). Stripped before match.
_INVISIBLE = re.compile(r"[​-‏‪-‮⁠-⁤﻿]")
_WHITESPACE = re.compile(r"\s+")

# Each rule is (code, description, pattern). Keeping them separate rather than
# one alternation costs nothing at this input size and tells the audit log
# exactly which rule fired.
_RULES: tuple[tuple[BlockReason, str, re.Pattern[str]], ...] = (
    (
        BlockReason.INSTRUCTION_OVERRIDE,
        "asks the model to discard its instructions",
        re.compile(
            r"\b(ignore|disregard|forget|override|bypass)\b[^.]{0,40}?"
            r"\b(previous|prior|above|earlier|preceding|initial|original|system|all)\b"
            r"[^.]{0,20}?\b(instruction|prompt|rule|direction|command|context|guideline)",
            re.IGNORECASE,
        ),
    ),
    (
        BlockReason.ROLE_HIJACK,
        "attempts to reassign the model's role",
        re.compile(
            r"\b(you\s+are\s+now|from\s+now\s+on\s+you|act\s+as\s+(?:a\s+|an\s+)?"
            r"(?:dan|jailbroken|unrestricted|unfiltered)|pretend\s+(?:to\s+be|you)"
            r"|developer\s+mode|jailbreak)\b",
            re.IGNORECASE,
        ),
    ),
    (
        BlockReason.PROMPT_EXTRACTION,
        "tries to exfiltrate the system prompt",
        # The object must be addressed to the model ("your instructions", "the
        # system prompt", "instructions you were given"). Matching a bare
        # "instructions" also blocks "show me the instructions manual", which is
        # an ordinary question a user is entitled to ask.
        re.compile(
            r"\b(?:reveal|show|print|repeat|output|display|tell\s+me|what\s+(?:is|are))\b"
            r"[^.]{0,30}?\b(?:"
            r"your\s+(?:system\s+|initial\s+|original\s+)?"
            r"(?:prompt|instructions?|rules?|guidelines?)"
            r"|(?:the\s+)?(?:system|initial|original)\s+prompt"
            r"|prompt\s+template"
            r"|instructions?\s+(?:that\s+)?you\s+(?:were\s+)?(?:given|received)"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        BlockReason.DELIMITER_INJECTION,
        "injects chat or template control tokens",
        re.compile(
            r"(<\|[^|>]{1,40}\|>|\[/?(?:INST|SYS|SYSTEM)\]|###\s*(?:system|instruction)\b"
            r"|<\s*/?\s*(?:system|assistant)\s*>)",
            re.IGNORECASE,
        ),
    ),
    # Starter coverage for Devanagari and Tamil. These match the literal verbs
    # for "ignore"/"forget" next to "instruction"; they are deliberately narrow
    # and want a native-speaker review pass before being relied on in
    # production. See the README's known limitations.
    (
        BlockReason.INSTRUCTION_OVERRIDE,
        "Hindi instruction-override phrasing",
        re.compile(r"(अनदेखा|नज़रअंदाज|भूल\s*जाओ|भूल\s*जाइए)[^।]{0,30}?(निर्देश|आदेश)"),
    ),
    (
        BlockReason.INSTRUCTION_OVERRIDE,
        "Tamil instruction-override phrasing",
        re.compile(r"(புறக்கணி|மறந்து)[^.]{0,30}?(அறிவுறுத்தல|கட்டளை)"),
    ),
)


@dataclass(frozen=True)
class InputVerdict:
    """Result of the pre-retrieval check."""

    allowed: bool
    normalized_query: str
    latency_ms: float
    reason: BlockReason | None = None
    description: str = ""
    matched_text: str = ""

    @property
    def code(self) -> str:
        return self.reason.value if self.reason else "ALLOW"


@dataclass(frozen=True)
class GroundingVerdict:
    """Result of the post-retrieval check."""

    allowed: bool
    top_score: float
    threshold: float
    latency_ms: float
    kept: int = 0
    message: str = ""

    @property
    def code(self) -> str:
        return "ALLOW" if self.allowed else BlockReason.UNGROUNDED.value


def normalize_query(text: str) -> str:
    """Fold away the cheap evasions before matching.

    NFKC maps compatibility forms (fullwidth latin, styled unicode letters) onto
    their plain equivalents, so a query written in lookalike characters matches
    the same rules as one written normally. Invisible characters are removed
    outright rather than replaced with a space, since they are inserted *inside*
    words specifically to split them.
    """
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", text)
    folded = _INVISIBLE.sub("", folded)
    return _WHITESPACE.sub(" ", folded).strip()


def check_input(query: str) -> InputVerdict:
    """Pre-retrieval filter. Runs before any embedding or network call."""
    started = time.perf_counter()

    normalized = normalize_query(query)

    if not normalized:
        return InputVerdict(
            allowed=False,
            normalized_query="",
            latency_ms=(time.perf_counter() - started) * 1000,
            reason=BlockReason.EMPTY_QUERY,
            description="query is empty",
        )

    if len(normalized) > MAX_QUERY_CHARS:
        return InputVerdict(
            allowed=False,
            normalized_query=normalized[:MAX_QUERY_CHARS],
            latency_ms=(time.perf_counter() - started) * 1000,
            reason=BlockReason.QUERY_TOO_LONG,
            description=f"query exceeds {MAX_QUERY_CHARS} characters",
        )

    for reason, description, pattern in _RULES:
        match = pattern.search(normalized)
        if match:
            return InputVerdict(
                allowed=False,
                normalized_query=normalized,
                latency_ms=(time.perf_counter() - started) * 1000,
                reason=reason,
                description=description,
                matched_text=match.group(0)[:120],
            )

    return InputVerdict(
        allowed=True,
        normalized_query=normalized,
        latency_ms=(time.perf_counter() - started) * 1000,
    )


def check_grounding(
    scores: Sequence[float], threshold: float = SIMILARITY_THRESHOLD
) -> GroundingVerdict:
    """Post-retrieval grounding check on cosine similarities.

    Scores must come from L2-normalized vectors searched with inner product, so
    they are true cosine similarities on [-1, 1] and comparable to `threshold`.
    """
    started = time.perf_counter()

    top_score = max(scores) if scores else 0.0
    kept = sum(1 for score in scores if score >= threshold)
    allowed = top_score >= threshold

    return GroundingVerdict(
        allowed=allowed,
        top_score=float(top_score),
        threshold=threshold,
        latency_ms=(time.perf_counter() - started) * 1000,
        kept=kept,
        message="" if allowed else UNGROUNDED_MESSAGE,
    )


@dataclass
class AuditEntry:
    """One row in the guardrails audit log shown in the UI."""

    stage: str  # "input" | "grounding"
    code: str
    allowed: bool
    latency_ms: float
    detail: str = ""
    query_preview: str = ""


def audit_input(verdict: InputVerdict, query: str) -> AuditEntry:
    return AuditEntry(
        stage="input",
        code=verdict.code,
        allowed=verdict.allowed,
        latency_ms=verdict.latency_ms,
        detail=verdict.description,
        query_preview=query[:80],
    )


def audit_grounding(verdict: GroundingVerdict, query: str) -> AuditEntry:
    return AuditEntry(
        stage="grounding",
        code=verdict.code,
        allowed=verdict.allowed,
        latency_ms=verdict.latency_ms,
        detail=f"top={verdict.top_score:.4f} threshold={verdict.threshold:.2f} kept={verdict.kept}",
        query_preview=query[:80],
    )
