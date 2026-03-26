Good. PRD 1 should be boring, concrete, and strict. Its job is not novelty. Its job is to create a reliable baseline you can trust later.

# PRD 1 — Baseline RAG Stack

## Project name

**Baseline RAG Infrastructure for Contamination-Aware Evaluation**

## Purpose

Build one or two strong, reproducible baseline RAG pipelines that will serve as the fixed reference systems for later contamination-focused evaluation and controller development.

This stage is not about inventing a new method. It is about establishing a trustworthy baseline stack, consistent logging, and a clean evaluation interface.

## Why this stage exists

Before designing a contamination-aware controller, we need to know:

* how a competent RAG system behaves on clean and noisy inputs,
* where it fails,
* what artifacts and logs are available for later control decisions,
* and whether the failure mode is visible in practice.

If this baseline is weak or inconsistent, later results will be hard to interpret.

---

# 1. Goals

## Primary goal

Implement a fixed, reproducible RAG pipeline that can answer questions from retrieved evidence and save all intermediate artifacts needed for later analysis.

## Secondary goals

* support comparison across one or two baseline variants,
* log retrieval outputs, rankings, prompts, answers, and confidence-related signals,
* create a clean interface for downstream benchmarking,
* make the stack modular so later controller insertion is easy.

## Non-goals

This stage will **not**:

* implement contamination scoring,
* implement subset selection,
* implement abstention policies beyond standard model behavior,
* tune for state-of-the-art leaderboard performance,
* build a full production service.

---

# 2. Success criteria

PRD 1 is complete if all of the following are true:

1. at least one baseline RAG system runs end-to-end on a benchmark dataset,
2. all retrieved passages, scores, prompts, and outputs are saved in structured form,
3. runs are reproducible from config files,
4. results can be evaluated automatically,
5. the codebase is modular enough that a controller can later be inserted between retrieval and generation,
6. a small qualitative error pack is produced showing representative successes and failures.

Preferred:
7. two baseline variants exist, such as:

* dense or hybrid retrieval without reranking
* dense or hybrid retrieval with reranking

---

# 3. User stories

## Researcher story

As a researcher, I want a reliable baseline RAG system so that later improvements can be attributed to the contamination-aware controller rather than baseline instability.

## Engineering story

As an engineer, I want all intermediate outputs logged in a standard format so that I can later compute contamination features without rewriting the core stack.

## Evaluation story

As an evaluator, I want reproducible configs and standardized outputs so I can compare multiple systems on the same examples.

---

# 4. System requirements

## 4.1 High-level pipeline

The baseline pipeline should follow this sequence:

1. load query
2. retrieve top-k passages
3. optionally rerank retrieved passages
4. construct generation context
5. generate answer
6. save all intermediate artifacts
7. run evaluation

## 4.2 Required components

### Query input layer

Takes benchmark query examples and normalizes them into a common schema.

### Retriever

At least one retriever is required.

Recommended options:

* dense retriever
* hybrid lexical + dense retriever

### Optional reranker

A reranker should be easy to toggle on or off.

### Generator

One fixed answer-generation model with a stable prompt template.

### Evaluation module

Computes answer correctness and stores metrics.

### Logging layer

Stores intermediate artifacts in structured form.

---

# 5. Baseline variants

## Required minimum baseline

### Baseline A

**Standard RAG**

* retrieve top-k
* optionally pass in ranked order directly
* answer from full retrieved context

## Preferred second baseline

### Baseline B

**Standard RAG + reranking**

* retrieve top-k from retriever
* rerank top-k
* answer from reranked full context

## Optional third baseline

### Baseline C

**Reduced-context RAG**

* retrieve top-k
* answer from top-1 or top-2 only

This is useful later as a simple baseline for whether “less context” helps.

---

# 6. Functional specs

## 6.1 Input schema

Each query example should have a normalized structure like:

```json id="8m7nna"
{
  "example_id": "ex_0001",
  "question": "When was Company X acquired by Company Y?",
  "gold_answer": "2019",
  "metadata": {
    "dataset": "dataset_name",
    "split": "dev"
  }
}
```

## 6.2 Retrieval output schema

The retriever must return at least:

```json id="pa6ca1"
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

## 6.3 Reranker output schema

If reranking is enabled:

```json id="5y9f0m"
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

## 6.4 Prompt construction

The system must construct prompts from:

* question
* retrieved passages in order
* fixed system instructions
* fixed answer formatting instructions

Prompt construction must be deterministic given config and retrieved order.

## 6.5 Generation output schema

```json id="22n1dh"
{
  "example_id": "ex_0001",
  "final_answer": "Company Y acquired Company X in 2019.",
  "raw_model_output": "...",
  "used_passage_ids": ["p3", "p1", "p4", "p2"],
  "generation_metadata": {
    "model_name": "model_x",
    "temperature": 0.0
  }
}
```

## 6.6 Evaluation output schema

```json id="7zxxjj"
{
  "example_id": "ex_0001",
  "is_correct": true,
  "score": 1.0,
  "normalized_prediction": "2019",
  "normalized_gold": "2019"
}
```

---

# 7. Logging requirements

This stage lives or dies on logging. Save more than you think you need.

## Required logs per example

* query text
* gold answer
* retrieved passages
* raw retrieval scores
* reranked passages, if applicable
* rerank scores, if applicable
* full generation prompt
* final answer
* raw model output
* model metadata
* evaluation result
* run config hash or ID

## Preferred logs

* token counts
* latency per stage
* truncated-context indicators
* prompt length
* generation probability or confidence proxy if available

## File outputs

At minimum:

* `predictions.jsonl`
* `retrievals.jsonl`
* `prompts.jsonl`
* `evaluations.jsonl`
* `run_config.yaml`
* `summary_metrics.json`

---

# 8. Configuration requirements

All major behaviors must be config-driven.

## Required config fields

* dataset name
* split
* retriever type
* retriever params
* top-k
* reranker on/off
* reranker params
* generator model
* prompt template ID
* temperature
* max tokens
* output directory
* random seed

## Example config

```yaml id="7985w6"
run_name: baseline_rag_v1
dataset: benchmark_dev
split: dev

retriever:
  type: hybrid
  top_k: 8

reranker:
  enabled: true
  model: bge_reranker
  top_k_after_rerank: 5

generator:
  model_name: gpt_4_1_or_equivalent
  temperature: 0.0
  max_tokens: 200

prompt:
  template_id: qa_grounded_v1

logging:
  save_prompts: true
  save_raw_outputs: true

seed: 42
```

---

# 9. Evaluation requirements

## Required evaluation capabilities

The system must support automatic scoring for at least one QA-style benchmark.

### Required metrics

* exact match or normalized match
* accuracy
* optional token-level F1 if applicable

### Preferred metrics

* answer length
* citation or grounding heuristics if available
* agreement between raw answer and normalized extracted answer

## Evaluation constraints

Scoring should be separated from generation so that predictions can be rescored later without rerunning the model.

---

# 10. Architecture requirements

## Design principle

The code should be organized so the future controller can be inserted **between retrieval/reranking and generation**.

### Desired interface

Something like:

```python
retrieved = retrieve(query)
reranked = rerank(retrieved, query)
context_bundle = build_context_bundle(query, reranked)
answer = generate(context_bundle)
```

Later, this should become:

```python
retrieved = retrieve(query)
reranked = rerank(retrieved, query)
control_decision = controller(query, reranked)
context_bundle = build_context_bundle(query, control_decision.selected_passages)
answer = generate(context_bundle)
```

That means passage objects and ranking metadata need to stay clean and reusable.

---

# 11. Recommended code modules

Suggested structure:

```text
rag_project/
  configs/
  data/
  src/
    data_loading/
    retrieval/
    reranking/
    prompting/
    generation/
    evaluation/
    logging/
    pipelines/
    utils/
  scripts/
    run_baseline.py
    evaluate_predictions.py
    inspect_examples.py
  outputs/
```

## Module responsibilities

### `data_loading/`

Load benchmark datasets and normalize them.

### `retrieval/`

Retriever wrappers and retrieval utilities.

### `reranking/`

Reranker wrappers and ranking utilities.

### `prompting/`

Prompt templates and context formatting.

### `generation/`

LLM call wrapper and output parsing.

### `evaluation/`

Scoring and normalization functions.

### `logging/`

Structured JSONL writers, run metadata, summaries.

### `pipelines/`

End-to-end orchestration.

---

# 12. Implementation tasks

## Task group A: dataset ingestion

* implement dataset loader
* normalize examples to common schema
* write dataset preview script

## Task group B: retrieval

* implement retriever wrapper
* return top-k passages with scores
* save retrieval outputs

## Task group C: reranking

* implement reranker wrapper
* rerank retrieval outputs
* save reranked lists

## Task group D: prompt construction

* implement fixed grounded-QA prompt template
* format retrieved passages deterministically

## Task group E: generation

* implement answer generation wrapper
* save raw outputs and parsed answer

## Task group F: evaluation

* implement normalized answer scoring
* generate summary metrics

## Task group G: orchestration

* implement config-driven run script
* save all artifacts under run directory

## Task group H: inspection

* implement script that samples successes and failures
* export small error pack for manual review

---

# 13. Exit criteria

PRD 1 is done when:

1. `run_baseline.py` completes end-to-end on at least one benchmark split,
2. outputs are saved in structured files,
3. summary metrics are computed automatically,
4. two rerunnable configs work without code changes,
5. a qualitative pack of at least 25 examples is exported for inspection,
6. code paths are ready for later controller insertion.

Preferred:
7. Baseline A and Baseline B have both been run on the same split.

---

# 14. Risks and mitigations

## Risk 1

The baseline is too weak, so later controller gains are meaningless.

### Mitigation

Use a competent retriever and optionally reranker. Do not intentionally cripple the stack.

## Risk 2

The logging is incomplete, so later contamination analysis becomes painful.

### Mitigation

Over-log by default.

## Risk 3

The code is too entangled, making controller insertion messy.

### Mitigation

Keep passage objects, ranking stages, and generation interfaces modular.

## Risk 4

Evaluation is too benchmark-specific.

### Mitigation

Normalize data inputs and predictions to a common schema.

---

# 15. Open decisions to settle now

These should be fixed before implementation starts:

1. Which retriever for v1?
2. Will hybrid retrieval be default?
3. Which reranker for v1?
4. Which generator model for v1?
5. What top-k and post-rerank k values will be default?
6. Which initial dataset or benchmark split will be used?
7. What confidence proxy, if any, should be logged now for future use?

---

# 16. Recommended default choices

For a pragmatic v1, I’d lean toward:

* **Retriever:** hybrid if easy, otherwise strong dense retrieval
* **Reranker:** yes, one standard reranker
* **Generator:** one stable high-quality model with temperature 0
* **Top-k retrieval:** 8 to 10
* **Top-k after rerank:** 4 to 6
* **Prompt:** simple grounded QA prompt with instruction to answer from provided evidence only

The point is not squeezing maximum benchmark performance. The point is building a baseline people will respect.

---

# 17. Deliverables

At the end of PRD 1, you should have:

* baseline RAG pipeline code
* run configs
* structured outputs
* automatic evaluation
* qualitative error pack
* short README explaining how to rerun experiments

