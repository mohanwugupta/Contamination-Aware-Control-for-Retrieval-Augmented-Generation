"""RED tests for prompt templates (PRD 1 §12).

Prompt families:
- A: single-answer grounded QA
- B: multi-answer grounded QA
- C: unknown-compatible grounded QA

All prompts must be deterministic and not contain hidden heuristics.
"""

import pytest


class TestPromptTemplates:
    """Tests for prompt template rendering."""

    def test_single_answer_prompt_renders(self):
        from rag_baseline.prompts.templates import render_prompt

        prompt = render_prompt(
            question="When was Michael Jordan born?",
            context_text="[1] Michael Jordan was born in 1963.",
            answer_mode="single",
        )
        assert "Michael Jordan" in prompt
        assert "1963" in prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_multi_answer_prompt_renders(self):
        from rag_baseline.prompts.templates import render_prompt

        prompt = render_prompt(
            question="When was Michael Jordan born?",
            context_text="[1] MJ basketball born 1963. [2] MJ actor born 1987.",
            answer_mode="multi",
        )
        assert "Michael Jordan" in prompt
        assert isinstance(prompt, str)

    def test_unknown_compatible_prompt_renders(self):
        from rag_baseline.prompts.templates import render_prompt

        prompt = render_prompt(
            question="What is the population of Atlantis?",
            context_text="[1] Atlantis is a mythological city.",
            answer_mode="unknown_or_abstain",
        )
        assert "Atlantis" in prompt
        assert isinstance(prompt, str)

    def test_prompt_is_deterministic(self):
        from rag_baseline.prompts.templates import render_prompt

        p1 = render_prompt(question="Q?", context_text="C", answer_mode="single")
        p2 = render_prompt(question="Q?", context_text="C", answer_mode="single")
        assert p1 == p2

    def test_prompt_no_context_for_llm_only(self):
        from rag_baseline.prompts.templates import render_prompt

        prompt = render_prompt(
            question="When was Michael Jordan born?",
            context_text="",
            answer_mode="single",
        )
        assert "Michael Jordan" in prompt
        assert isinstance(prompt, str)

    def test_multi_answer_prompt_instructs_multiple_answers(self):
        from rag_baseline.prompts.templates import render_prompt

        prompt = render_prompt(
            question="Q?",
            context_text="C",
            answer_mode="multi",
        )
        # The multi-answer prompt should instruct the model to provide all answers
        assert "all" in prompt.lower() or "multiple" in prompt.lower() or "each" in prompt.lower()

    def test_unknown_prompt_instructs_unknown(self):
        from rag_baseline.prompts.templates import render_prompt

        prompt = render_prompt(
            question="Q?",
            context_text="C",
            answer_mode="unknown_or_abstain",
        )
        # Should mention unknown / unanswerable / not enough info
        prompt_lower = prompt.lower()
        assert any(w in prompt_lower for w in ["unknown", "unanswerable", "not enough", "cannot"])
