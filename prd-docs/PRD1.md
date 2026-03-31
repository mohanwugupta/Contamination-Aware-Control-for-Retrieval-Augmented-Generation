# PRD 1 — Baseline RAG Systems and Benchmark Harness

## Project name

**Baseline RAG Systems for Contamination-Aware Evaluation**

## Purpose

Build a small set of strong, reproducible baseline RAG systems and a benchmark harness that will serve as the fixed reference point for all later contamination-focused work.

This stage is not about inventing a new method.
It is about building the baseline systems that later work must beat.

PRD 1 should answer a narrow question:

> What happens when a competent but standard RAG stack is run across ordinary factual QA, ambiguity-heavy QA, and conflicting-evidence QA, with enough logging and evaluation structure to support later controller insertion?

## Why this stage exists

Before building a contamination-aware controller, we need a baseline stack that tells us:

* how ordinary RAG behaves on clean retrieval,
* how it degrades under ambiguity and conflicting evidence,
* which failure modes are already visible without any special controller,
* what artifacts are available between retrieval and generation,
* and which baseline comparisons later stages must preserve.

If PRD 1 is weak, later gains will be uninterpretable.

If PRD 1 is underspecified, later evaluation will drift.

---

# 1. Core thesis of this stage

PRD 1 is not “build one generic RAG pipeline.”

PRD 1 is:

1. build a **baseline family** of standard RAG variants,
2. run them on a **benchmark ladder** that becomes progressively more aligned with the project’s target failure mode,
3. freeze their interfaces, outputs, and logging so later controller work can plug into the exact same harness.

That means PRD 1 must own both:

* the **systems**
* and the **evaluation harness**

but not the contamination-aware controller itself.

---

# 2. Goals

## Primary goal

Implement a reproducible baseline RAG evaluation stack that can run multiple standard baseline systems across multiple benchmark types and save all intermediate artifacts needed for downstream contamination analysis.

## Secondary goals

* compare at least three baseline variants with fixed prompts and configs,
* establish a benchmark ladder from sanity-check QA to ambiguity and conflicting-evidence QA,
* support both single-answer and multi-answer tasks,
* log retrieval, reranking, prompt construction, answer generation, and evaluation artifacts in structured form,
* make controller insertion easy later by keeping a clean separation between:

  * retrieval,
  * reranking,
  * context assembly,
  * generation,
  * evaluation.

## Non-goals

This stage will **not**:

* implement contamination scoring,
* implement subset selection,
* implement abstention policies beyond baseline model behavior,
* implement the project’s proposed controller,
* tune for leaderboard performance at all costs,
* build a production API or serving layer,
* evaluate enterprise black-box systems as a required milestone.

---

# 3. What PRD 1 must produce

PRD 1 must produce three things:

## A. A baseline system matrix

At minimum:

* **Baseline 0 — LLM-only / no retrieval control**
* **Baseline A — Vanilla RAG**
* **Baseline B — Hybrid RAG**
* **Baseline C — Hybrid RAG + reranker**
* **Baseline D — Reduced-context RAG**

Not every one of these must be treated as a headline system in every table, but all except Baseline 0 should exist in the harness, and at least A/B/C must be run in the main PRD 1 deliverable.

## B. A benchmark ladder

PRD 1 should not jump straight to the hardest benchmark only.

It should use a staged benchmark ladder:

* **Tier 0 — NQ or comparable ordinary factual QA sanity benchmark**
* **Tier 1 — AmbigDocs**
* **Tier 2 — FaithEval**
* **Tier 3 — RAMDocs**

## C. A normalized evaluation interface

PRD 1 must support:

* single-answer factual QA
* multi-answer ambiguous QA
* unknown / unanswerable outputs
* context-faithfulness style evaluation where applicable

This is mandatory. A single EM-only evaluator is not enough.

---

# 4. Benchmark ladder

## Tier 0 — Sanity benchmark

### Dataset

**NQ** or another standard short-answer factual QA benchmark.

### Purpose

Use this tier to verify:

* the stack runs end-to-end,
* retrieval and reranking are not broken,
* stronger retrieval variants are not obviously degenerate,
* clean factual QA performance is reasonable.

### Expected role

This is not the project’s main scientific benchmark.
It is the baseline sanity check.

## Tier 1 — Core project benchmark

### Dataset

**AmbigDocs**

### Purpose

This is the first benchmark that directly reflects the project’s target failure mode.

Use it to test whether baseline RAG systems can handle:

* same-name entity ambiguity,
* multiple correct answers for a single surface-form query,
* the need to disambiguate entities explicitly rather than merge them.

### Expected role

This is the **core benchmark for PRD 1**.

If the baseline harness cannot run AmbigDocs cleanly, PRD 1 is not done.

## Tier 2 — Robustness benchmark

### Dataset

**FaithEval**

### Purpose

Use it to test whether baseline systems remain faithful to context under:

* unanswerable contexts,
* inconsistent contexts,
* counterfactual contexts.

### Expected role

This is the main check that stronger ambiguity handling is not achieved by simply becoming sloppier or less context-faithful elsewhere.

## Tier 3 — Stretch benchmark

### Dataset

**RAMDocs**

### Purpose

Use it to stress-test the baseline systems under mixed conflict:

* ambiguity,
* misinformation,
* noise,
* uneven support across answers.

### Expected role

This is the hardest benchmark in PRD 1 and should be treated as the forward-looking stress test.

It is preferred for MVP-plus and required if resources allow.

---

# 5. Baseline system matrix

## Baseline 0 — LLM-only control

### Description

Answer the question without retrieval.

### Purpose

Provides a non-RAG control so later results can distinguish:

* retrieval helping,
* retrieval hurting,
* retrieval introducing new failure modes.

### Required?

Optional for minimal implementation, but strongly recommended.

---

## Baseline A — Vanilla RAG

### Description

* retrieve top-k passages
* no reranking
* concatenate retrieved passages in retrieval order
* answer from full retrieved context

### Purpose

This is the weakest serious RAG baseline.

It gives the project a plain reference point.

### Required?

Yes.

---

## Baseline B — Hybrid RAG

### Description

* retrieve top-k passages using hybrid lexical + dense retrieval
* no reranking
* answer from full retrieved context

### Purpose

This should be the minimum “respectable” baseline.

### Required?

Yes.

---

## Baseline C — Hybrid RAG + reranker

### Description

* hybrid retrieve top-k
* rerank retrieved passages
* answer from reranked full context

### Purpose

This is the main strong baseline for PRD 1.

### Required?

Yes.

---

## Baseline D — Reduced-context RAG

### Description

* retrieve and optionally rerank as above
* answer from top-1 or top-2 passages only

### Purpose

This is a diagnostic baseline.

It tests whether some gains later could come from “just use less context.”

### Required?

Yes in code, preferred in experiments.

---

## Optional external comparator

### Example

Enterprise or managed system such as Bedrock/Kendra.

### Purpose

Useful as an external point of comparison, but not required for the first reproducible project-owned baseline stack.

### Required?

No.

---

# 6. Success criteria

PRD 1 is complete if all of the following are true:

1. Baseline A, B, and C run end-to-end on at least one split each of Tier 0 and Tier 1 benchmarks.
2. At least one reduced-context variant runs on the same split as a full-context baseline.
3. All retrievals, rankings, prompts, answers, and evaluation outputs are saved in structured form.
4. Runs are reproducible from config files without code edits.
5. The system supports both single-answer and multi-answer evaluation modes.
6. A controller could later be inserted between retrieval and generation without rewriting the stack.
7. A qualitative inspection pack is exported showing representative:

   * clean successes,
   * ambiguity failures,
   * conflicting-evidence failures.

Preferred:

8. FaithEval is integrated and run.
9. RAMDocs is integrated and run.
10. LLM-only control is included.
11. Two retrieval settings and two generation-context settings are compared on the same benchmark slice.

---

# 7. User stories

## Researcher story

As a researcher, I want a strong but standard baseline family so later gains can be attributed to the contamination-aware controller rather than baseline weakness.

## Engineering story

As an engineer, I want standardized artifacts at each stage so I can add contamination scoring later without rewriting the baseline stack.

## Evaluation story

As an evaluator, I want dataset-specific scoring adapters behind one common interface so I can compare systems fairly across ordinary QA, ambiguity QA, and conflicting-evidence QA.

## Analysis story

As an analyst, I want enough metadata and logging to inspect whether a failure came from:

* retrieval quality,
* ambiguity,
* conflicting evidence,
* too much context,
* or the generator itself.

---

# 8. High-level pipeline

Every baseline should follow the same stage structure:

1. load benchmark example
2. normalize example into common schema
3. retrieve candidate passages
4. optionally rerank candidate passages
5. choose passage subset according to baseline definition
6. build generation prompt deterministically
7. generate answer
8. parse answer into task-specific output format
9. evaluate answer with dataset-specific scorer
10. save all intermediate artifacts

The point is that all baseline variants differ only in a few controlled places:

* retrieval type
* reranking on/off
* number of passages used
* answer formatting mode

---

# 9. Required components

## 9.1 Query input layer

Normalizes benchmark examples into a common schema.

Must support:

* single gold answer
* multiple gold answers
* unanswerable / abstain-compatible targets
* dataset-specific metadata

## 9.2 Retriever layer

At least two retriever settings should be supported:

* dense retrieval
* hybrid lexical + dense retrieval

If only one can be implemented at first, make hybrid the preferred default.

## 9.3 Reranker layer

A reranker must be easy to toggle on or off by config.

## 9.4 Context assembly layer

Must support:

* full retrieved set
* reranked full retrieved set
* reduced-context top-1
* reduced-context top-2

Deterministic formatting is required.

## 9.5 Generator layer

One fixed answer-generation model for the main PRD 1 runs.

Requirements:

* fixed prompt template family
* temperature 0 or near-greedy decoding
* stable answer formatting instructions

## 9.6 Output parser layer

Must support at least three answer modes:

* `single`
* `multi`
* `unknown_or_abstain`

## 9.7 Evaluation layer

Must dispatch to dataset-specific scorers while presenting a uniform outer interface.

---

# 10. Required normalized schema

## 10.1 Input schema

```json
{
  "example_id": "ex_0001",
  "question": "In which year was Michael Jordan born?",
  "task_type": "multi_answer_qa",
  "gold": {
    "single_answer": null,
    "multi_answers": ["    "multi_answers": ["1963", "1956"],
    "unknown_allowed": false
  },
  "metadata": {
    "dataset": "ambigdocs",
    "split": "dev"
  }
}
```

## 10.2 Retrieval output schema

```json
{
  "example_id": "ex_0001",
  "retrieved_passages": [
    {
      "passage_id": "p1",
      "text": "...",
      "source": "doc_123",
      "retrieval_score": 13.2,
      "rank": 1
    }
  ]
}
```

## 10.3 Reranker output schema

```json
{
  "example_id": "ex_0001",
  "reranked_passages": [
    {
      "passage_id": "p3",
      "text": "...",
      "source": "doc_999",
      "retrieval_score": 11.8,
      "rerank_score": 0.87,
      "rank_after_rerank": 1
    }
  ]
}
```

## 10.4 Prompt record schema

```json
{
  "example_id": "ex_0001",
  "baseline_name": "hybrid_rerank_full",
  "answer_mode": "multi",
  "used_passage_ids": ["p3", "p1", "p4", "p2"],
  "prompt_text": "...",
  "prompt_metadata": {
    "model_name": "generator_x",
    "temperature": 0.0,
    "max_context_passages": 4
  }
}
```

## 10.5 Generation output schema

```json
{
  "example_id": "ex_0001",
  "raw_model_output": "...",
  "parsed_output": {
    "single_answer": null,
    "multi_answers": ["1963", "1956"],
    "unknown": false
  }
}
```

## 10.6 Evaluation output schema

```json
{
  "example_id": "ex_0001",
  "dataset": "ambigdocs",
  "baseline_name": "hybrid_rerank_full",
  "metrics": {
    "exact_match": null,
    "normalized_match": null,
    "multi_answer_score": 1.0,
    "answer_category": "complete"
  }
}
```

---

# 11. Dataset-specific output contracts

## 11.1 Ordinary factual QA contract

For NQ-like tasks:

* parse a single short answer
* compute normalized exact match or equivalent

## 11.2 Ambiguity QA contract

For AmbigDocs-like tasks:

* allow multiple correct answers
* preserve explicit entity disambiguation when the benchmark requires it
* support categories such as:

  * complete
  * partial
  * ambiguous
  * merged
  * no answer

## 11.3 Context-faithfulness contract

For FaithEval-like tasks:

* support outputs that may be:

  * correct,
  * incorrect,
  * unknown,
  * or task-specific depending on the benchmark adapter
* preserve ability to inspect answerability separately from correctness

## 11.4 Mixed-conflict contract

For RAMDocs-like tasks:

* allow multiple correct answers when ambiguity is genuine
* allow unknown when context is unusable
* do not force a single-answer parser on inherently mixed examples

---

# 12. Prompting requirements

## Core prompting rule

PRD 1 prompts should be simple and fixed.

Do not build benchmark-specific reasoning tricks into the main baseline prompts.

The baseline should be competent, but standard.

## Required prompt families

### Prompt family A — single-answer grounded QA

Use for NQ-like settings.

### Prompt family B — multi-answer grounded QA

Use for ambiguity-heavy and mixed-conflict settings.

Required behavior:

* if there are multiple clearly supported answers, provide all of them
* if the context does not support an answer, say unknown
* do not invent support outside the provided documents

### Prompt family C — unknown-compatible grounded QA

Use for unanswerable or strongly context-faithfulness-oriented settings.

## Prompt design constraints

* deterministic formatting
* no hidden benchmark-specific heuristics
* no controller-like passage filtering inside the prompt
* no test-time self-review loop as part of the baseline

---

# 13. Evaluation requirements

## Required metrics across the harness

Every run should report:

* accuracy or task-correctness metric
* unknown / abstention rate where applicable
* answerable-only accuracy where applicable
* per-dataset breakdown
* per-baseline breakdown

## Required benchmark-specific reporting

### NQ-like

* normalized EM
* accuracy

### AmbigDocs-like

* complete / partial / ambiguous / merged / no-answer breakdown
* main accuracy summary derived from dataset scorer

### FaithEval-like

* task-wise reporting for:

  * unanswerable
  * inconsistent
  * counterfactual
* overall summary

### RAMDocs-like

* overall exact match or task metric
* breakdown by conflict subtype if available from adapter

---

# 14. Logging requirements

This stage lives or dies on logging.

## Required logs per example

* normalized input example
* raw retrieved passages
* retrieval scores
* reranked passages, if any
* rerank scores, if any
* final used passage set
* full prompt text
* raw model output
* parsed output
* evaluation result
* run config ID or hash

## Preferred logs

* token counts by stage
* latency by stage
* truncated-context indicator
* passage count used
* answer mode used
* parser warnings
* confidence proxy if available

## Required file outputs

At minimum:

* `inputs.jsonl`
* `retrievals.jsonl`
* `reranks.jsonl`
* `prompts.jsonl`
* `predictions.jsonl`
* `evaluations.jsonl`
* `run_config.yaml`
* `summary_metrics.json`

---

# 15. Configuration requirements

All major behavior must be config-driven.

## Required config fields

* dataset
* split
* retriever_type
* reranker_enabled
* generator_model
* prompt_family
* top_k_retrieval
* top_k_after_rerank
* context_strategy
* answer_mode
* output_dir
* random_seed

## Required invariant

Two runs with the same config and seed should be rerunnable without code edits.

---

# 16. Recommended default choices

For pragmatic v1:

* **Retriever default:** hybrid retrieval
* **Second retriever setting:** dense-only
* **Reranker default:** enabled for strong baseline
* **Generator:** one stable high-quality model
* **Decoding:** temperature 0
* **Top-k retrieval:** 8 to 10
* **Top-k after rerank:** 4 to 6
* **Reduced-context diagnostic:** top-2
* **Tier 0 benchmark:** NQ
* **Tier 1 benchmark:** AmbigDocs
* **Tier 2 benchmark:** FaithEval
* **Tier 3 benchmark:** RAMDocs if resources allow

---

# 17. Suggested implementation phases

## Phase 1

Implement:

* common schema
* Vanilla RAG
* Hybrid RAG
* NQ adapter
* AmbigDocs adapter
* deterministic prompting
* structured logging

## Phase 2

Add:

* reranker
* reduced-context baseline
* FaithEval adapter
* multi-answer parser improvements

## Phase 3

Add:

* RAMDocs adapter
* stronger reporting
* inspection scripts
* benchmark comparison pack

---

# 18. Implementation tasks

## Task group A — schemas and adapters

* finalize common example schema
* implement NQ adapter
* implement AmbigDocs adapter
* implement FaithEval adapter
* implement RAMDocs adapter

## Task group B — retrieval

* implement dense retrieval wrapper
* implement hybrid retrieval wrapper
* standardize retrieval artifact format

## Task group C — reranking

* implement reranker wrapper
* standardize rerank artifact format
* add reranker toggle in config

## Task group D — context assembly

* implement full-context assembly
* implement reduced-context assembly
* ensure deterministic ordering and formatting

## Task group E — prompting and generation

* implement single-answer prompt family
* implement multi-answer prompt family
* implement unknown-compatible prompt family
* implement generation wrapper
* save raw and parsed outputs

## Task group F — parsing and evaluation

* implement single-answer parser
* implement multi-answer parser
* implement unknown parser
* implement dataset-specific scorers
* export summary metrics

## Task group G — orchestration

* implement config-driven run script
* implement run directory structure
* save all structured artifacts

## Task group H — inspection

* implement sampling script for representative cases
* export qualitative pack for manual review

---

# 19. Exit criteria

PRD 1 is done when:

1. Baseline A, B, and C can be run from config without code edits.
2. NQ and AmbigDocs are both supported and evaluated automatically.
3. At least one reduced-context baseline exists and runs.
4. Structured artifacts are saved for retrieval, prompt, prediction, and evaluation stages.
5. Multi-answer outputs are supported in the evaluation harness.
6. A qualitative inspection pack with at least 25 examples is exported.
7. The code path clearly supports later controller insertion between retrieval and generation.

Preferred:

8. FaithEval is integrated and run.
9. RAMDocs is integrated and run.
10. LLM-only control is included in the same harness.
11. A short README explains how to reproduce the main PRD 1 runs.

---

# 20. Risks and mitigations

## Risk 1

The baseline is too weak, so later controller gains are meaningless.

### Mitigation

Use one strong standard baseline as the anchor:
hybrid retrieval plus reranking plus stable prompting.

## Risk 2

The baseline is too benchmark-specific.

### Mitigation

Keep one common outer schema and one common run harness, but allow dataset-specific parsers and scorers.

## Risk 3

The harness assumes one-answer QA and breaks on ambiguity benchmarks.

### Mitigation

Make answer mode explicit in the schema and parser layer from the start.

## Risk 4

Reduced-context gains are confused with controller gains later.

### Mitigation

Include reduced-context as a first-class baseline now.

## Risk 5

PRD 1 drifts into controller design.

### Mitigation

Do not implement contamination scoring, subset selection, or abstention routing here.

## Risk 6

Logging is insufficient for later contamination analysis.

### Mitigation

Over-log by default and save passage identities and ordering.

---

# 21. Open decisions to settle now

These should be fixed before implementation starts:

1. Which generator model is the single fixed model for v1?
2. Is hybrid retrieval mandatory in phase 1 or phase 2?
3. Which reranker is the v1 default?
4. Which Tier 0 benchmark split will be the initial sanity run?
5. Which AmbigDocs slice will be the first mandatory run?
6. Whether FaithEval enters phase 1 or phase 2
7. Whether RAMDocs is in-scope for MVP or MVP-plus
8. Which confidence proxy, if any, should already be logged for later use

---

# 22. Deliverables

At the end of PRD 1, you should have:

* baseline RAG system code
* multiple baseline configs
* benchmark adapters
* structured outputs
* automatic evaluation
* qualitative inspection pack
* short reproduction README

The point is not just “a RAG pipeline exists.”

The point is:

> the project now has a baseline matrix and benchmark harness solid enough that later controller claims can be judged fairly.
