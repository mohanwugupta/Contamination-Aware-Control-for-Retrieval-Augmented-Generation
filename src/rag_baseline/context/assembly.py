"""Context assembly layer (PRD 1 §9.4).

Assembles retrieved (or reranked) passages into a formatted context
string for the generator. Supports full, reduced, and none strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag_baseline.schemas.rerank import RerankedPassage
from rag_baseline.schemas.retrieval import RetrievedPassage


@dataclass
class AssembledContext:
    """Result of context assembly."""

    passages: list[RetrievedPassage | RerankedPassage]
    passage_ids: list[str]
    formatted_text: str


def assemble_context(
    passages: list[RetrievedPassage | RerankedPassage],
    strategy: str,
    max_passages: int | None = None,
) -> AssembledContext:
    """Assemble passages into formatted context.

    Args:
        passages: Ordered list of passages (retrieval or rerank order).
        strategy: One of 'full', 'reduced', 'none'.
        max_passages: Maximum number of passages to include (for reduced).

    Returns:
        AssembledContext with passage list and formatted text.
    """
    if strategy == "none" or not passages:
        return AssembledContext(passages=[], passage_ids=[], formatted_text="")

    if strategy == "reduced" and max_passages is not None:
        passages = passages[:max_passages]

    passage_ids = [p.passage_id for p in passages]

    # Deterministic formatting: numbered passages
    lines = []
    for i, p in enumerate(passages, start=1):
        lines.append(f"[Document {i}] {p.text}")

    formatted_text = "\n\n".join(lines)

    return AssembledContext(
        passages=passages,
        passage_ids=passage_ids,
        formatted_text=formatted_text,
    )
