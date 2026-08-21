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
    UNSAFE_CONTENT = "UNSAFE_CONTENT"
    PII_DETECTED = "PII_DETECTED"
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
    # --- unsafe content ---
    #
    # These match a request for ACTIONABLE instructions, not a mention of the
    # topic. A retrieval corpus legitimately contains passages about explosives
    # in mining or the pharmacology of overdose, and a user is entitled to ask
    # about them. "How do I build a bomb" is a different act from "what is TNT",
    # and only the first is matched here.
    (
        BlockReason.UNSAFE_CONTENT,
        "requests actionable instructions for causing physical harm",
        re.compile(
            r"\b(how\s+(?:do|can|would)\s+(?:i|you|we)|steps?\s+to|instructions?\s+for"
            r"|guide\s+to|teach\s+me\s+to|best\s+way\s+to)\b[^.?]{0,40}?"
            # Stems with \w* so inflections match too: "synthesizing",
            # "making", "manufacturing" are the same request as the infinitive.
            r"\b(?:build|mak|construct|synthesi[sz]|manufactur|assembl|obtain|acquir)\w*\b"
            r"[^.?]{0,30}?\b(bomb|explosive|ied|napalm|nerve\s+agent|sarin|ricin|anthrax"
            r"|meth(?:amphetamine)?|fentanyl|untraceable\s+(?:gun|weapon)|ghost\s+gun)\b",
            re.IGNORECASE,
        ),
    ),
    (
        BlockReason.UNSAFE_CONTENT,
        "requests instructions for harming a person",
        re.compile(
            r"\b(how\s+(?:do|can|to)\s*(?:i|you)?|best\s+way\s+to|steps?\s+to)\b"
            r"[^.?]{0,30}?\b(kill|murder|poison|strangle|stab|assassinate)\b"
            r"[^.?]{0,20}?\b(some\s?one|a\s+person|people|him|her|them"
            # Named relations only. A bare "my <word>" also matches "kill my
            # weeds", which is gardening, not violence.
            r"|my\s+(?:wife|husband|partner|boss|neighbou?r|friend|family|father|mother"
            r"|brother|sister|colleague|teacher|roommate))\b",
            re.IGNORECASE,
        ),
    ),
    (
        BlockReason.UNSAFE_CONTENT,
        "requests self-harm or suicide method information",
        re.compile(
            r"\b(how\s+(?:do|can|to)\s*(?:i)?|best|painless|easiest|quickest)\b"
            r"[^.?]{0,25}?\b(kill\s+myself|commit\s+suicide|end\s+my\s+life"
            r"|hang\s+myself|overdose\s+on)\b",
            re.IGNORECASE,
        ),
    ),
    (
        BlockReason.UNSAFE_CONTENT,
        "requests material sexualizing minors",
        re.compile(
            r"\b(child|minor|underage|preteen|teen)\b[^.?]{0,20}?"
            r"\b(porn|sexual|nude|naked|erotic)\b"
            r"|\b(porn|sexual|nude|erotic)\b[^.?]{0,20}?\b(child|minor|underage)\b",
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


# --- PII -------------------------------------------------------------------
#
# Blocked because the query is forwarded to third-party APIs (Sarvam, Groq).
# Refusing here keeps identifiers out of somebody else's logs, which is a
# stronger protection than any downstream policy we do not control.
#
# Digit sequences are validated, not merely matched: an unvalidated 16-digit
# pattern flags order numbers and document ids, and blocking ordinary questions
# is its own failure.
_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email address", re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    # Indian mobile numbers: 10 digits starting 6-9, optional +91.
    ("phone number", re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")),
    ("credit card number", re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")),
    ("Aadhaar number", re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")),
    ("US social security number", re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")),
)


def _luhn_valid(digits: str) -> bool:
    """Luhn checksum, so only plausible card numbers are treated as cards."""
    total = 0
    parity = len(digits) % 2
    for position, character in enumerate(digits):
        value = int(character)
        if position % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def detect_pii(text: str) -> str | None:
    """Return a description of the first PII found, or None."""
    for label, pattern in _PII_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        if label == "credit card number":
            digits = re.sub(r"\D", "", match.group(0))
            # Verify length and checksum; otherwise it is just a long number.
            if not (13 <= len(digits) <= 19 and _luhn_valid(digits)):
                continue
        return label
    return None


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

    pii_label = detect_pii(normalized)
    if pii_label:
        return InputVerdict(
            allowed=False,
            normalized_query=normalized,
            latency_ms=(time.perf_counter() - started) * 1000,
            reason=BlockReason.PII_DETECTED,
            description=f"query contains what looks like a {pii_label}",
            # The matched value is deliberately NOT recorded: writing it into
            # the audit log would leak the identifier we just refused to send
            # to a third party.
            matched_text="[redacted]",
        )

    return InputVerdict(
        allowed=True,
        normalized_query=normalized,
        latency_ms=(time.perf_counter() - started) * 1000,
    )


# --- small talk ------------------------------------------------------------
#
# "hi" is not a question, but it is the first thing almost anyone types. Sent
# down the retrieval path it embeds, matches a passage about the Japanese kana
# は at 0.7036 -- above the grounding threshold -- and reaches the model, which
# spends a second and real tokens correctly concluding the passage does not
# answer it. The user's first impression is then a red refusal to a greeting.
#
# Answering here instead costs a regex match. It is not a guardrail in the
# safety sense: nothing is being blocked, the input simply has a better answer
# than retrieval can give.
#
# Every pattern is anchored to the whole string. "hello" is small talk;
# "hello, what is inflation" is a question with a greeting attached, and must
# still go to retrieval.
_SMALL_TALK: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "greeting",
        re.compile(
            r"^(?:hi|hii+|hey+|hello+|yo|namaste|namaskar|vanakkam|hola"
            r"|good\s*(?:morning|afternoon|evening|day)"
            r"|नमस्ते|नमस्कार|हाय|हैलो|வணக்கம்|ஹாய்)"
            r"[\s!.?,]*$",
            re.IGNORECASE,
        ),
    ),
    (
        "howareyou",
        re.compile(
            r"^(?:how\s*(?:are|r)\s*(?:you|u)(?:\s*doing)?|what'?s\s*up|sup"
            r"|कैसे\s*हो|क्या\s*हाल\s*है|எப்படி\s*இருக்கிறீர்கள்)"
            r"[\s!.?,]*$",
            re.IGNORECASE,
        ),
    ),
    (
        "thanks",
        re.compile(
            r"^(?:thanks?|thank\s*(?:you|u)|thx|ty|dhanyavaad|shukriya"
            r"|धन्यवाद|शुक्रिया|நன்றி)[\s!.?,]*$",
            re.IGNORECASE,
        ),
    ),
    (
        "farewell",
        re.compile(
            r"^(?:bye+|goodbye|see\s*(?:you|ya)|alvida|अलविदा|பிரியாவிடை)"
            r"[\s!.?,]*$",
            re.IGNORECASE,
        ),
    ),
    (
        "capability",
        re.compile(
            r"^(?:who\s*(?:are|r)\s*(?:you|u)|what\s*(?:are|r)\s*(?:you|u)"
            r"|what\s*can\s*(?:you|u)\s*do|help|what\s*is\s*this"
            r"|तुम\s*कौन\s*हो|நீங்கள்\s*யார்)[\s!.?,]*$",
            re.IGNORECASE,
        ),
    ),
)

# Replies carry a nudge toward what the system can actually do, because a bare
# "hello" back leaves the user exactly as stuck as the refusal did.
_SMALL_TALK_REPLIES: dict[str, dict[str, str]] = {
    "greeting": {
        "en": "Hello. Ask me a question and I'll answer from the indexed passages — try \"what is a corporation\" or \"what causes diabetes\".",
        "hi": "नमस्ते। कोई सवाल पूछिए, मैं अनुक्रमित अंशों से उत्तर दूंगा — जैसे \"निगम क्या है\"।",
        "ta": "வணக்கம். ஒரு கேள்வி கேளுங்கள், அட்டவணைப்படுத்தப்பட்ட பத்திகளிலிருந்து பதிலளிக்கிறேன் — எடுத்துக்காட்டாக \"நிறுவனம் என்றால் என்ன\".",
    },
    "howareyou": {
        "en": "Running fine. Ask me something from the corpus and I'll answer from retrieved passages.",
        "hi": "सब ठीक है। कोई सवाल पूछिए, मैं अनुक्रमित अंशों से उत्तर दूंगा।",
        "ta": "நன்றாக இயங்குகிறேன். ஒரு கேள்வி கேளுங்கள், பத்திகளிலிருந்து பதிலளிக்கிறேன்.",
    },
    "thanks": {
        "en": "You're welcome. Ask another question whenever you like.",
        "hi": "आपका स्वागत है। जब चाहें दूसरा सवाल पूछिए।",
        "ta": "வரவேற்கிறேன். வேறு கேள்வி இருந்தால் கேளுங்கள்.",
    },
    "farewell": {
        "en": "Goodbye.",
        "hi": "अलविदा।",
        "ta": "பிரியாவிடை.",
    },
    "capability": {
        "en": "I answer questions from an indexed corpus of passages, by voice or text, in English, Hindi or Tamil. I only answer from what was retrieved — if nothing relevant is found, I say so rather than guess.",
        "hi": "मैं अनुक्रमित अंशों से सवालों के जवाब देता हूँ — आवाज़ या टेक्स्ट से, अंग्रेज़ी, हिंदी या तमिल में। जो नहीं मिला, उसका अनुमान नहीं लगाता।",
        "ta": "அட்டவணைப்படுத்தப்பட்ட பத்திகளிலிருந்து கேள்விகளுக்கு பதிலளிக்கிறேன் — குரல் அல்லது உரை மூலம், ஆங்கிலம், இந்தி அல்லது தமிழில். கிடைக்காததை ஊகிக்க மாட்டேன்.",
    },
}


def detect_small_talk(text: str, language: str = "en") -> tuple[str, str] | None:
    """Return (kind, reply) when the input is a greeting rather than a question.

    Runs on the normalized query, so the same invisible-character and unicode
    folding that protects the injection rules applies here too.
    """
    for kind, pattern in _SMALL_TALK:
        if pattern.match(text):
            replies = _SMALL_TALK_REPLIES[kind]
            return kind, replies.get(language) or replies["en"]
    return None


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
