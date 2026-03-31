"""RED tests for the output parser (PRD 1 §9.6).

Parser must support three answer modes:
- single
- multi
- unknown_or_abstain
"""

import pytest


class TestOutputParser:
    """Tests for parsing raw model outputs into structured form."""

    def test_parse_single_answer(self):
        from rag_baseline.parsing.output_parser import parse_output

        raw = "The answer is 1963."
        result = parse_output(raw, answer_mode="single")
        assert result.single_answer is not None
        assert result.unknown is False

    def test_parse_multi_answer(self):
        from rag_baseline.parsing.output_parser import parse_output

        raw = "Answers: [\"1963\", \"1956\"]"
        result = parse_output(raw, answer_mode="multi")
        assert result.multi_answers is not None
        assert len(result.multi_answers) >= 1

    def test_parse_unknown_answer(self):
        from rag_baseline.parsing.output_parser import parse_output

        raw = "UNKNOWN"
        result = parse_output(raw, answer_mode="unknown_or_abstain")
        assert result.unknown is True

    def test_parse_single_strips_whitespace(self):
        from rag_baseline.parsing.output_parser import parse_output

        raw = "  1963  \n"
        result = parse_output(raw, answer_mode="single")
        assert result.single_answer == "1963"

    def test_parse_multi_from_numbered_list(self):
        from rag_baseline.parsing.output_parser import parse_output

        raw = "1. 1963\n2. 1956"
        result = parse_output(raw, answer_mode="multi")
        assert result.multi_answers is not None
        assert "1963" in result.multi_answers
        assert "1956" in result.multi_answers

    def test_parse_multi_from_json_array(self):
        from rag_baseline.parsing.output_parser import parse_output

        raw = '["Paris", "London"]'
        result = parse_output(raw, answer_mode="multi")
        assert result.multi_answers is not None
        assert "Paris" in result.multi_answers

    def test_parse_returns_parsed_output_type(self):
        from rag_baseline.parsing.output_parser import parse_output
        from rag_baseline.schemas.generation import ParsedOutput

        result = parse_output("answer", answer_mode="single")
        assert isinstance(result, ParsedOutput)

    def test_parse_unknown_keywords(self):
        """Various unknown-like responses should be detected."""
        from rag_baseline.parsing.output_parser import parse_output

        for raw in ["UNKNOWN", "I don't know", "unanswerable", "Not enough information"]:
            result = parse_output(raw, answer_mode="unknown_or_abstain")
            assert result.unknown is True, f"Failed to detect unknown for: {raw!r}"

    def test_parse_non_unknown_in_unknown_mode(self):
        from rag_baseline.parsing.output_parser import parse_output

        raw = "1963"
        result = parse_output(raw, answer_mode="unknown_or_abstain")
        assert result.unknown is False
        assert result.single_answer == "1963"
