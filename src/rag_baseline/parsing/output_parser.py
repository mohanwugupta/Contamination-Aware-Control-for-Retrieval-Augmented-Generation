"""Output parser (PRD 1 §9.6).

Parses raw model output into structured ParsedOutput.
Supports single, multi, and unknown_or_abstain answer modes.
"""

from __future__ import annotations

import json
import re

from rag_baseline.schemas.generation import ParsedOutput


# Keywords that indicate the model is abstaining / saying unknown
UNKNOWN_KEYWORDS = [
    "unknown",
    "i don't know",
    "i do not know",
    "unanswerable",
    "not enough information",
    "cannot determine",
    "cannot answer",
    "insufficient information",
    "no answer",
    "not answerable",
]


def _is_unknown(text: str) -> bool:
    """Check if text indicates an unknown / abstain response."""
    text_lower = text.strip().lower()
    return any(kw in text_lower for kw in UNKNOWN_KEYWORDS)


def _parse_multi_answers(raw: str) -> list[str]:
    """Try to extract multiple answers from raw text.

    Supports:
    - JSON arrays: ["a", "b"]
    - Numbered lists: 1. a\\n2. b
    - Newline-separated answers
    """
    raw = raw.strip()

    # Try JSON array first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (json.JSONDecodeError, TypeError):
        pass

    # Try numbered list: "1. answer\n2. answer"
    numbered = re.findall(r"^\d+[\.\)]\s*(.+)$", raw, re.MULTILINE)
    if numbered:
        return [a.strip() for a in numbered if a.strip()]

    # Try bullet list: "- answer\n- answer"
    bullets = re.findall(r"^[-•*]\s*(.+)$", raw, re.MULTILINE)
    if bullets:
        return [a.strip() for a in bullets if a.strip()]

    # Fall back to newline-separated
    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    if len(lines) > 1:
        return lines

    # Single answer
    if raw.strip():
        return [raw.strip()]

    return []


def parse_output(raw: str, answer_mode: str) -> ParsedOutput:
    """Parse raw model output into structured form.

    Args:
        raw: Raw text from the model.
        answer_mode: One of 'single', 'multi', 'unknown_or_abstain'.

    Returns:
        ParsedOutput with appropriate fields populated.
    """
    raw = raw.strip()

    if answer_mode == "unknown_or_abstain":
        if _is_unknown(raw):
            return ParsedOutput(single_answer=None, multi_answers=None, unknown=True)
        else:
            # Not unknown — treat as single answer
            return ParsedOutput(single_answer=raw, multi_answers=None, unknown=False)

    elif answer_mode == "single":
        return ParsedOutput(single_answer=raw, multi_answers=None, unknown=False)

    elif answer_mode == "multi":
        answers = _parse_multi_answers(raw)
        return ParsedOutput(single_answer=None, multi_answers=answers, unknown=False)

    else:
        raise ValueError(f"Unknown answer_mode: {answer_mode}")
