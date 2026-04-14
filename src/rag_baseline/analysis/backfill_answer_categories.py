"""Backfill evaluations.jsonl files: relabel mislabelled 'no_answer' records.

Background
----------
Prior to the scorer fix (commit: 'wrong vs no_answer'), the multi-answer scorer
assigned ``answer_category = "no_answer"`` to TWO semantically distinct cases:

  1. Empty predictions  → model *abstained*.  Correct label: "no_answer".
  2. Non-empty predictions, recall = 0.0 → model *answered wrongly*.
     Correct label: "wrong".

This script re-labels case 2 in every saved ``evaluations.jsonl`` by
cross-referencing ``predictions.jsonl`` (same run directory).  It only touches
multi-answer datasets (ambigdocs, ramdocs).

Each evaluations.jsonl is rewritten in-place; a ``.bak`` copy is kept.

Usage
-----
    python src/rag_baseline/analysis/backfill_answer_categories.py
    # or with a custom outputs root:
    python src/rag_baseline/analysis/backfill_answer_categories.py --output-dir outputs
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

MULTI_ANSWER_DATASETS = {"ambigdocs", "ramdocs"}


def _load_predictions_index(predictions_path: Path) -> dict[str, list[str]]:
    """Return {example_id: [predicted_answer_strings]} from predictions.jsonl."""
    index: dict[str, list[str]] = {}
    with predictions_path.open() as fh:
        for line in fh:
            obj = json.loads(line)
            example_id = obj["example_id"]
            parsed = obj.get("parsed_output", {})
            multi = parsed.get("multi_answers") or []
            index[example_id] = multi
    return index


def backfill_run(run_dir: Path, dry_run: bool = False) -> dict[str, int]:
    """Backfill a single run directory.

    Returns a dict with counts: {relabelled, skipped_non_multi, total}.
    """
    evals_path = run_dir / "evaluations.jsonl"
    preds_path = run_dir / "predictions.jsonl"

    if not evals_path.exists() or not preds_path.exists():
        return {"relabelled": 0, "skipped_no_files": 1, "total": 0}

    # Load evaluations and check dataset type
    records: list[dict] = []
    with evals_path.open() as fh:
        for line in fh:
            records.append(json.loads(line))

    if not records:
        return {"relabelled": 0, "total": 0}

    dataset = records[0].get("dataset", "")
    if dataset not in MULTI_ANSWER_DATASETS:
        return {"relabelled": 0, "skipped_non_multi": len(records), "total": len(records)}

    # Build predictions index
    preds_index = _load_predictions_index(preds_path)

    # Relabel
    relabelled = 0
    for rec in records:
        if rec.get("metrics", {}).get("answer_category") != "no_answer":
            continue
        example_id = rec["example_id"]
        multi_answers = preds_index.get(example_id, [])
        if multi_answers:
            # Non-empty prediction with zero recall → was wrongly labelled "no_answer"
            rec["metrics"]["answer_category"] = "wrong"
            relabelled += 1

    if not dry_run and relabelled > 0:
        # Keep a backup
        backup_path = evals_path.with_suffix(".jsonl.bak")
        if not backup_path.exists():
            shutil.copy2(evals_path, backup_path)

        # Rewrite in-place
        with evals_path.open("w") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")

    return {"relabelled": relabelled, "total": len(records)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill 'wrong' vs 'no_answer' in evaluations.jsonl")
    parser.add_argument(
        "--output-dir", "-i", default="outputs",
        help="Root outputs directory containing run subdirectories (default: outputs)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would change without modifying any files"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: {output_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    total_relabelled = 0
    for run_dir in sorted(output_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if "smoke_test" in run_dir.name:
            continue
        stats = backfill_run(run_dir, dry_run=args.dry_run)
        n = stats.get("relabelled", 0)
        if n > 0 or stats.get("total", 0) > 0:
            tag = "[DRY RUN] " if args.dry_run else ""
            print(f"{tag}{run_dir.name}: relabelled {n}/{stats.get('total', 0)} records")
        total_relabelled += n

    action = "Would relabel" if args.dry_run else "Relabelled"
    print(f"\n{action} {total_relabelled} records total.")
    if args.dry_run:
        print("Re-run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
