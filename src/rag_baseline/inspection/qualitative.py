"""Qualitative inspection pack (PRD 1 §6 criterion 7, §18 Task Group H).

Samples representative examples from run artifacts and exports a
qualitative inspection pack for manual review.

Categories:
    - clean_success: correct answers on clean retrieval
    - ambiguity_failure: failures on ambiguous queries
    - conflicting_evidence_failure: failures on conflicting/counterfactual contexts
    - unknown_correct: correct abstentions on unanswerable contexts
    - other: uncategorized
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict


def categorize_artifacts(artifacts: list[dict]) -> dict[str, list[dict]]:
    """Group artifacts by their outcome category.

    Args:
        artifacts: List of artifact dicts, each must have a 'category' key.

    Returns:
        Dict mapping category name to list of artifacts in that category.
    """
    categories: dict[str, list[dict]] = defaultdict(list)
    for artifact in artifacts:
        cat = artifact.get("category", "other")
        categories[cat].append(artifact)
    return dict(categories)


def sample_inspection_pack(
    artifacts: list[dict],
    min_total: int = 25,
    seed: int | None = None,
) -> list[dict]:
    """Sample representative examples for qualitative inspection.

    Ensures coverage across all categories present in the artifacts.
    Tries to sample at least `min_per_category` from each, filling
    up to `min_total` total examples.

    Args:
        artifacts: List of artifact dicts with 'category' keys.
        min_total: Minimum total examples to include in the pack.
        seed: Random seed for reproducibility.

    Returns:
        List of sampled artifact dicts.
    """
    rng = random.Random(seed)

    categories = categorize_artifacts(artifacts)
    cat_names = sorted(categories.keys())

    if not cat_names:
        return []

    # Phase 1: guarantee min representation from each category
    min_per_cat = max(1, min_total // len(cat_names))
    sampled: list[dict] = []
    sampled_ids: set[str] = set()

    for cat in cat_names:
        pool = categories[cat]
        n = min(min_per_cat, len(pool))
        selected = rng.sample(pool, n)
        for item in selected:
            if item["example_id"] not in sampled_ids:
                sampled.append(item)
                sampled_ids.add(item["example_id"])

    # Phase 2: fill up to min_total from remaining
    if len(sampled) < min_total:
        remaining = [a for a in artifacts if a["example_id"] not in sampled_ids]
        rng.shuffle(remaining)
        for item in remaining:
            if len(sampled) >= min_total:
                break
            sampled.append(item)
            sampled_ids.add(item["example_id"])

    return sampled


def export_inspection_pack(pack: list[dict], output_path: str) -> None:
    """Export the inspection pack to JSONL and a summary JSON.

    Creates:
        - {output_path}: JSONL file with one example per line
        - {dir}/inspection_summary.json: summary with category counts

    Args:
        pack: List of artifact dicts to export.
        output_path: Path to the JSONL output file.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write JSONL
    with open(output_path, "w") as f:
        for item in pack:
            f.write(json.dumps(item, default=str) + "\n")

    # Write summary
    categories = categorize_artifacts(pack)
    summary = {
        "total_examples": len(pack),
        "category_counts": {cat: len(items) for cat, items in sorted(categories.items())},
        "categories": sorted(categories.keys()),
    }

    summary_path = os.path.join(output_dir, "inspection_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
