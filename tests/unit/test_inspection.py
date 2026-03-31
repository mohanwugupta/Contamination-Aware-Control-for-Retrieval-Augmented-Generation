"""RED tests for qualitative inspection pack (PRD 1 §6 criterion 7, §18 Task H).

The inspection module samples representative examples from run artifacts
and exports a qualitative pack showing:
    - clean successes
    - ambiguity failures
    - conflicting-evidence failures
    - unknown/abstention cases

Must produce ≥25 examples covering diverse outcome categories.
"""

import json
import os
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Sample evaluation artifacts for testing
# ---------------------------------------------------------------------------

def _make_artifact(
    example_id: str,
    question: str,
    prediction: str,
    gold_answer: str,
    is_correct: bool,
    dataset: str = "nq_open",
    baseline_name: str = "vanilla_rag",
    category: str = "clean_success",
    retrieved_passages: list[dict] | None = None,
    prompt_text: str = "Answer: ...",
) -> dict:
    """Build a mock run artifact dict."""
    return {
        "example_id": example_id,
        "question": question,
        "prediction": prediction,
        "gold_answer": gold_answer,
        "is_correct": is_correct,
        "dataset": dataset,
        "baseline_name": baseline_name,
        "category": category,
        "retrieved_passages": retrieved_passages or [],
        "prompt_text": prompt_text,
    }


SAMPLE_ARTIFACTS = [
    # Clean successes
    _make_artifact("nq_0", "Capital of France?", "Paris", "Paris", True, category="clean_success"),
    _make_artifact("nq_1", "Largest ocean?", "Pacific", "Pacific", True, category="clean_success"),
    _make_artifact("nq_2", "Speed of light?", "299792458 m/s", "299792458 m/s", True, category="clean_success"),
    _make_artifact("nq_3", "H2O is?", "water", "water", True, category="clean_success"),
    _make_artifact("nq_4", "Earth's star?", "Sun", "Sun", True, category="clean_success"),
    _make_artifact("nq_5", "Boiling point?", "100C", "100C", True, category="clean_success"),
    _make_artifact("nq_6", "Highest peak?", "Everest", "Everest", True, category="clean_success"),
    _make_artifact("nq_7", "Closest planet?", "Mercury", "Mercury", True, category="clean_success"),
    _make_artifact("nq_8", "DNA shape?", "double helix", "double helix", True, category="clean_success"),
    _make_artifact("nq_9", "Pi value?", "3.14159", "3.14159", True, category="clean_success"),
    # Ambiguity failures
    _make_artifact("ambig_0", "Who is Jordan?", "Michael Jordan", "Michael Jordan;Jordan (country)", False, dataset="ambigdocs", category="ambiguity_failure"),
    _make_artifact("ambig_1", "When was Georgia founded?", "1732", "1785;1732", False, dataset="ambigdocs", category="ambiguity_failure"),
    _make_artifact("ambig_2", "Capital of Springfield?", "unknown", "Springfield IL;Springfield MO", False, dataset="ambigdocs", category="ambiguity_failure"),
    _make_artifact("ambig_3", "Who wrote Mercury?", "Freddie Mercury", "planet;element;artist", False, dataset="ambigdocs", category="ambiguity_failure"),
    _make_artifact("ambig_4", "When did Apple start?", "1976", "1976;Apple Records 1968", False, dataset="ambigdocs", category="ambiguity_failure"),
    _make_artifact("ambig_5", "Who is Washington?", "George W.", "George;state;DC", False, dataset="ambigdocs", category="ambiguity_failure"),
    _make_artifact("ambig_6", "What is Python?", "snake", "snake;language", False, dataset="ambigdocs", category="ambiguity_failure"),
    _make_artifact("ambig_7", "What is Java?", "island", "island;language;coffee", False, dataset="ambigdocs", category="ambiguity_failure"),
    # Conflicting-evidence failures
    _make_artifact("faith_0", "Water freezes at?", "100C", "unknown", False, dataset="faitheval", category="conflicting_evidence_failure"),
    _make_artifact("faith_1", "Closest to Sun?", "Venus", "conflict", False, dataset="faitheval", category="conflicting_evidence_failure"),
    _make_artifact("faith_2", "Gravity constant?", "wrong", "unknown", False, dataset="faitheval", category="conflicting_evidence_failure"),
    _make_artifact("faith_3", "Moon material?", "marshmallows", "unknown", False, dataset="faitheval", category="conflicting_evidence_failure"),
    _make_artifact("faith_4", "Boiling at?", "50C", "conflict", False, dataset="faitheval", category="conflicting_evidence_failure"),
    # Unknown/abstention
    _make_artifact("un_0", "Unsupported question?", "unknown", "unknown", True, dataset="faitheval", category="unknown_correct"),
    _make_artifact("un_1", "No info available?", "unknown", "unknown", True, dataset="faitheval", category="unknown_correct"),
    _make_artifact("un_2", "Unanswerable?", "I don't know", "unknown", True, dataset="faitheval", category="unknown_correct"),
]


# ===================================================================
# InspectionPack: categorization
# ===================================================================


class TestCategorizeArtifacts:
    """Test artifact categorization into outcome buckets."""

    def test_categorize_returns_dict(self):
        from rag_baseline.inspection.qualitative import categorize_artifacts

        result = categorize_artifacts(SAMPLE_ARTIFACTS)
        assert isinstance(result, dict)

    def test_categorize_has_required_categories(self):
        from rag_baseline.inspection.qualitative import categorize_artifacts

        result = categorize_artifacts(SAMPLE_ARTIFACTS)
        assert "clean_success" in result
        assert "ambiguity_failure" in result
        assert "conflicting_evidence_failure" in result
        assert "unknown_correct" in result

    def test_categorize_clean_success_count(self):
        from rag_baseline.inspection.qualitative import categorize_artifacts

        result = categorize_artifacts(SAMPLE_ARTIFACTS)
        assert len(result["clean_success"]) == 10

    def test_categorize_ambiguity_failure_count(self):
        from rag_baseline.inspection.qualitative import categorize_artifacts

        result = categorize_artifacts(SAMPLE_ARTIFACTS)
        assert len(result["ambiguity_failure"]) == 8

    def test_categorize_conflicting_failure_count(self):
        from rag_baseline.inspection.qualitative import categorize_artifacts

        result = categorize_artifacts(SAMPLE_ARTIFACTS)
        assert len(result["conflicting_evidence_failure"]) == 5


# ===================================================================
# InspectionPack: sampling
# ===================================================================


class TestSampleInspectionPack:
    """Test the sampling function that selects representative examples."""

    def test_sample_returns_list(self):
        from rag_baseline.inspection.qualitative import sample_inspection_pack

        pack = sample_inspection_pack(SAMPLE_ARTIFACTS, min_total=25)
        assert isinstance(pack, list)

    def test_sample_meets_minimum(self):
        from rag_baseline.inspection.qualitative import sample_inspection_pack

        pack = sample_inspection_pack(SAMPLE_ARTIFACTS, min_total=25)
        assert len(pack) >= 25

    def test_sample_covers_all_categories(self):
        from rag_baseline.inspection.qualitative import sample_inspection_pack

        pack = sample_inspection_pack(SAMPLE_ARTIFACTS, min_total=25)
        categories = {item["category"] for item in pack}
        assert "clean_success" in categories
        assert "ambiguity_failure" in categories
        assert "conflicting_evidence_failure" in categories

    def test_sample_respects_seed(self):
        from rag_baseline.inspection.qualitative import sample_inspection_pack

        pack1 = sample_inspection_pack(SAMPLE_ARTIFACTS, min_total=10, seed=42)
        pack2 = sample_inspection_pack(SAMPLE_ARTIFACTS, min_total=10, seed=42)
        ids1 = [p["example_id"] for p in pack1]
        ids2 = [p["example_id"] for p in pack2]
        assert ids1 == ids2

    def test_sample_different_seeds_may_differ(self):
        from rag_baseline.inspection.qualitative import sample_inspection_pack

        pack1 = sample_inspection_pack(SAMPLE_ARTIFACTS, min_total=10, seed=42)
        pack2 = sample_inspection_pack(SAMPLE_ARTIFACTS, min_total=10, seed=99)
        ids1 = [p["example_id"] for p in pack1]
        ids2 = [p["example_id"] for p in pack2]
        # May or may not differ, but at least check both return valid results
        assert len(ids1) >= 10
        assert len(ids2) >= 10


# ===================================================================
# InspectionPack: export to JSONL
# ===================================================================


class TestExportInspectionPack:
    """Test exporting the inspection pack to JSONL."""

    def test_export_creates_file(self):
        from rag_baseline.inspection.qualitative import (
            export_inspection_pack,
            sample_inspection_pack,
        )

        pack = sample_inspection_pack(SAMPLE_ARTIFACTS, min_total=25)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "inspection_pack.jsonl")
            export_inspection_pack(pack, output_path)
            assert os.path.exists(output_path)

    def test_export_jsonl_line_count(self):
        from rag_baseline.inspection.qualitative import (
            export_inspection_pack,
            sample_inspection_pack,
        )

        pack = sample_inspection_pack(SAMPLE_ARTIFACTS, min_total=25)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "inspection_pack.jsonl")
            export_inspection_pack(pack, output_path)
            with open(output_path) as f:
                lines = f.readlines()
            assert len(lines) >= 25

    def test_export_lines_are_valid_json(self):
        from rag_baseline.inspection.qualitative import (
            export_inspection_pack,
            sample_inspection_pack,
        )

        pack = sample_inspection_pack(SAMPLE_ARTIFACTS, min_total=25)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "inspection_pack.jsonl")
            export_inspection_pack(pack, output_path)
            with open(output_path) as f:
                for line in f:
                    obj = json.loads(line)
                    assert "example_id" in obj
                    assert "category" in obj
                    assert "question" in obj

    def test_export_includes_summary_file(self):
        from rag_baseline.inspection.qualitative import (
            export_inspection_pack,
            sample_inspection_pack,
        )

        pack = sample_inspection_pack(SAMPLE_ARTIFACTS, min_total=25)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "inspection_pack.jsonl")
            export_inspection_pack(pack, output_path)
            summary_path = os.path.join(tmpdir, "inspection_summary.json")
            assert os.path.exists(summary_path)

    def test_export_summary_has_counts(self):
        from rag_baseline.inspection.qualitative import (
            export_inspection_pack,
            sample_inspection_pack,
        )

        pack = sample_inspection_pack(SAMPLE_ARTIFACTS, min_total=25)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "inspection_pack.jsonl")
            export_inspection_pack(pack, output_path)
            summary_path = os.path.join(tmpdir, "inspection_summary.json")
            with open(summary_path) as f:
                summary = json.load(f)
            assert "total_examples" in summary
            assert "category_counts" in summary
            assert summary["total_examples"] >= 25
