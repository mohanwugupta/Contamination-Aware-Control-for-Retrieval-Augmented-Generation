"""Prompt templates (PRD 1 §12).

Three prompt families:
- A: single-answer grounded QA
- B: multi-answer grounded QA
- C: unknown-compatible grounded QA

All prompts are deterministic and contain no hidden heuristics.
"""

from __future__ import annotations


# --- Prompt family A: single-answer grounded QA ---
SINGLE_ANSWER_TEMPLATE = """\
Answer the following question based on the provided documents. \
Give a short, direct answer.

{context_block}\
Question: {question}

Answer:"""

SINGLE_ANSWER_NO_CONTEXT_TEMPLATE = """\
Answer the following question. Give a short, direct answer.

Question: {question}

Answer:"""

# --- Prompt family B: multi-answer grounded QA ---
MULTI_ANSWER_TEMPLATE = """\
Answer the following question based on the provided documents. \
If there are multiple correct answers (for example, referring to different entities \
with the same name), provide all of them. List each answer on a separate line.

{context_block}\
Question: {question}

Answers:"""

MULTI_ANSWER_NO_CONTEXT_TEMPLATE = """\
Answer the following question. \
If there are multiple correct answers, provide all of them. \
List each answer on a separate line.

Question: {question}

Answers:"""

# --- Prompt family C: unknown-compatible grounded QA ---
UNKNOWN_COMPATIBLE_TEMPLATE = """\
Answer the following question based only on the provided documents. \
If the documents do not contain enough information to answer the question, \
respond with "UNKNOWN". Do not guess or use information outside the documents.

{context_block}\
Question: {question}

Answer:"""

UNKNOWN_COMPATIBLE_NO_CONTEXT_TEMPLATE = """\
Answer the following question. \
If you cannot determine the answer, respond with "UNKNOWN".

Question: {question}

Answer:"""


def render_prompt(
    question: str,
    context_text: str,
    answer_mode: str,
) -> str:
    """Render a prompt from a template.

    Args:
        question: The question to answer.
        context_text: The formatted context string (empty for LLM-only).
        answer_mode: One of 'single', 'multi', 'unknown_or_abstain'.

    Returns:
        The rendered prompt string.
    """
    has_context = bool(context_text.strip())

    if has_context:
        context_block = f"Documents:\n{context_text}\n\n"
    else:
        context_block = ""

    if answer_mode == "single":
        if has_context:
            return SINGLE_ANSWER_TEMPLATE.format(
                context_block=context_block, question=question,
            )
        else:
            return SINGLE_ANSWER_NO_CONTEXT_TEMPLATE.format(question=question)

    elif answer_mode == "multi":
        if has_context:
            return MULTI_ANSWER_TEMPLATE.format(
                context_block=context_block, question=question,
            )
        else:
            return MULTI_ANSWER_NO_CONTEXT_TEMPLATE.format(question=question)

    elif answer_mode == "unknown_or_abstain":
        if has_context:
            return UNKNOWN_COMPATIBLE_TEMPLATE.format(
                context_block=context_block, question=question,
            )
        else:
            return UNKNOWN_COMPATIBLE_NO_CONTEXT_TEMPLATE.format(question=question)

    else:
        raise ValueError(f"Unknown answer_mode: {answer_mode}")
