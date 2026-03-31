# PRD 1 — Baseline RAG Stack

## Project name

**Baseline RAG Infrastructure for Contamination-Aware Evaluation**

## Purpose

Build one or two strong, reproducible baseline RAG pipelines that will serve as the fixed reference systems for later contamination-focused evaluation and controller development.

This stage is not about inventing a new method. It is about establishing a trustworthy baseline stack, consistent logging, strong test discipline, and a clean evaluation interface.

The implementation will use **FlashRAG** as the underlying research harness for retrieval, reranking, generation, and benchmark execution where practical. However, FlashRAG will be treated as an execution framework, not the source of truth for this project’s benchmark semantics, schemas, logging contracts, or evaluation definitions. ([arXiv][1])

## Why this stage exists

Before designing a contamination-aware controller, we need to know:

* how a competent RAG system behaves on clean and noisy inputs,
* where it fails,
* what artifacts and logs are available for later control decisions,
* whether the failure mode is visible in practice,
* and whether the baseline itself is stable enough to support later scientific claims.

If this baseline is weak, inconsistent, or under-tested, later controller results will be hard to interpret.

---

# 1. Goals

## Primary goal

Implement a fixed, reproducible RAG pipeline that can answer questions from retrieved evidence and save all intermediate artifacts needed for later analysis.

## Secondary goals

* support comparison across one or two baseline variants,
* log retrieval outputs, rankings, prompts, answers, and confidence-related signals,
* create a clean interface for downstream benchmarking,
* make the stack modular so later controller insertion is easy,
* ensure the baseline is robust enough to run reliably on the cluster,
* ensure core behavior is protected by tests.

## Non-goals

This stage will **not**:

* implement contamination scoring,
* implement subset selection,
* implement abstention policies beyond standard model behavior,
* tune for state-of-the-art leaderboard performance,
* build a full production service,
* let framework defaults silently define benchmark behavior.

---

# 2. Development philosophy

## 2.1 Test-driven development is required

This project will use **test-driven development** as a default engineering method.

New functionality should be developed using a strict:

**red → green → refactor**

loop:

1. write a failing test for the intended behavior,
2. implement the smallest change needed to make the test pass,
3. refactor only after the relevant tests are green.

Tests are part of the product, not a cleanup step.

## 2.2 What must be protected by tests

The following must be protected by tests:

* benchmark input schema normalization,
* retrieval and reranking output schemas,
* prompt construction behavior,
* generation output parsing,
* evaluation logic,
* logging and artifact writing,
* reproducibility guarantees,
* cluster execution entrypoints,
* controller insertion compatibility.

## 2.3 Engineering rules

* Do not weaken tests to accommodate buggy behavior.
* Every bug fix must add a regression test that would have caught it.
* Prefer small, explicit, verifiable changes.
* Prefer adapters over deep framework rewrites.
* Prefer deterministic fixtures and local smoke tests before cluster-scale execution.

---

# 3. Execution environment

## 3.1 Cluster-first execution

This baseline stack is intended to run primarily on the lab’s **cluster**, not just on a laptop.

The codebase must therefore be:

* CLI-first,
* config-driven,
* batchable,
* shardable,
* resumable where possible,
* non-interactive by default,
* explicit about cache, output, and scratch paths.

## 3.2 Local execution still matters

The project must also support:

* a tiny local smoke-test configuration,
* fixture-based integration tests,
* fast debugging without cluster submission.

Cluster is the default for real runs. Local is required for development safety.

---

# 4. Success criteria

PRD 1 is complete if all of the following are true:

1. at least one baseline RAG system runs end-to-end on a benchmark dataset using FlashRAG as the underlying research harness,
2. all retrieved passages, scores, prompts, and outputs are saved in structured **project-defined** form,
3. runs are reproducible from frozen config files and recorded framework versions,
4. results can be evaluated automatically without rerunning generation,
5. the codebase is modular enough that a controller can later be inserted between retrieval/reranking and generation,
6. a small qualitative error pack is produced showing representative successes and failures,
7. the system runs both:

   * as a local smoke-test configuration, and
   * as a cluster-ready batch configuration,
8. major modules are covered by tests written under a red → green → refactor workflow.

Preferred:

9. two baseline variants exist, such as:

   * dense or hybrid retrieval without reranking
   * dense or hybrid retrieval with reranking

10. every bug fixed during implementation adds a regression test.

---

# 5. User stories

## Researcher story

As a researcher, I want a reliable baseline RAG system so that later improvements can be attributed to the contamination-aware controller rather than baseline instability.

## Engineering story

As an engineer, I want all intermediate outputs logged in a standard format so that I can later compute contamination features without rewriting the core stack.

## Evaluation story

As an evaluator, I want reproducible configs and standardized outputs so I can compare multiple systems on the same examples.

## Infrastructure story

As a cluster user, I want runs to launch from CLI and config files, save complete manifests, and fail in inspectable ways so that large experiments are manageable.

---

# 6. Framework and system requirements

## 6.1 Framework stance

Use **FlashRAG** as the baseline research harness because it is designed for modular RAG experimentation and benchmark comparison. ([arXiv][1])

However:

* FlashRAG does **not** define the project’s benchmark semantics.
* FlashRAG does **not** define the project’s logging schema.
* FlashRAG does **not** define the project’s evaluation contract.
* FlashRAG does **not** define the future controller interface.

The project owns those interfaces.

## 6.2 Framework usage rules

* Pin FlashRAG by commit hash or release tag for reported runs.
* Save FlashRAG version metadata in every run manifest.
* Override framework defaults explicitly when they affect retrieval, reranking, prompt construction, truncation, generation settings, or evaluation.
* If FlashRAG objects do not match project schemas, write adapters. Do not change project schemas to fit framework internals.

## 6.3 High-level pipeline

The baseline pipeline should follow this sequence:

1. load query
2. normalize query into common project schema
3. retrieve top-k passages
4. optionally rerank retrieved passages
5. construct generation context
6. generate answer
7. save all intermediate artifacts
8. run evaluation
9. save run manifest and summary outputs

## 6.4 Required components

### Query input layer

Takes benchmark query examples and normalizes them into a common schema.

### Framework adapter layer

Wraps FlashRAG components and maps them to project-defined interfaces and schemas.

Responsibilities:

* normalize input examples into FlashRAG-compatible format,
* convert retrieval outputs into project retrieval schema,
* convert reranking outputs into project rerank schema,
* capture prompt and generation metadata,
* preserve example IDs and provenance.

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

### Run manifest / provenance layer

Stores:

* config,
* git commit,
* FlashRAG version,
* model identifiers,
* dataset version,
* seed,
* cluster job metadata,
* environment information,
* timestamped output paths.

---

# 7. Baseline variants

## Required minimum baseline

### Baseline A — Standard RAG

* retrieve top-k
* optionally pass in ranked order directly
* answer from full retrieved context

## Preferred second baseline

### Baseline B — Standard RAG + reranking

* retrieve top-k from retriever
* rerank top-k
* answer from reranked full context

## Optional third baseline

### Baseline C — Reduced-context RAG

* retrieve top-k
* answer from top-1 or top-2 only

This is useful later as a simple baseline for whether “less context” helps.

---

# 8. Functional specs

## 8.1 Input schema

Each query example should have a normalized structure like:

```json
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

## 8.2 Retrieval output schema

The retriever must return at least:

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

## 8.3 Reranker output schema

If reranking is enabled:

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

## 8.4 Prompt construction

The system must construct prompts from:

* question
* retrieved passages in order
* fixed system instructions
* fixed answer formatting instructions

Prompt construction must be deterministic given config and retrieved order.

## 8.5 Generation output schema

```json
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

## 8.6 Evaluation output schema

```json
{
  "example_id": "ex_0001",
  "is_correct": true,
  "score": 1.0,
  "normalized_prediction": "2019",
  "normalized_gold": "2019"
}
```

---

# 9. Logging requirements

This stage lives or dies on logging. Save more than you think you need.

## 9.1 Required logs per example

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

## 9.2 Required logs per run

* project git commit hash
* FlashRAG version / commit or release tag
* cluster job ID if applicable
* hostname / execution environment label
* resolved config file
* dataset version or data manifest
* retrieval index identifier
* reranker identifier
* generator identifier
* seed
* timestamp
* output directory

## 9.3 Preferred logs

* token counts
* latency per stage
* truncated-context indicators
* prompt length
* generation probability or confidence proxy if available
* stage-specific wall-clock times
* GPU/CPU device info
* cache directory used
* retries / failures by stage
* dry-run indicator
* shard index or job-array index

## 9.4 File outputs

At minimum:

* `predictions.jsonl`
* `retrievals.jsonl`
* `prompts.jsonl`
* `evaluations.jsonl`
* `run_config.yaml`
* `run_manifest.json`
* `summary_metrics.json`

---

# 10. Configuration requirements

All major behaviors must be config-driven.

## 10.1 Required config fields

* framework name and version pin
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
* local vs cluster execution mode
* cache directory
* scratch directory
* output root
* shard ID / job array parameters if used
* dry-run mode
* number of workers
* cluster resource profile
* retry policy
* test fixture mode

## 10.2 Example config

```yaml
run_name: baseline_rag_v1

framework:
  name: flashrag
  version_pin: "<commit-or-tag>"

execution:
  mode: cluster
  dry_run: false
  cache_dir: /scratch/$USER/flashrag_cache
  scratch_dir: /scratch/$USER/rag_tmp
  output_root: /scratch/$USER/rag_outputs
  num_workers: 4
  shard_id: 0
  num_shards: 1
  resource_profile: standard_gpu
  retry_policy: none
  fixture_mode: false

dataset:
  name: benchmark_dev
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

# 11. Evaluation requirements

## 11.1 Required evaluation capabilities

The system must support automatic scoring for at least one QA-style benchmark.

### Required metrics

* exact match or normalized match
* accuracy
* optional token-level F1 if applicable

### Preferred metrics

* answer length
* citation or grounding heuristics if available
* agreement between raw answer and normalized extracted answer

## 11.2 Evaluation constraints

* Scoring should be separated from generation so predictions can be rescored later without rerunning the model.
* Saved artifacts must be sufficient to rerun evaluation independently.
* Evaluation code must be test-covered.

---

# 12. Architecture requirements

## 12.1 Design principle

The code should be organized so the future controller can be inserted **between retrieval/reranking and generation**.

### Desired interface

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

## 12.2 Controller-compatibility requirement

The baseline must preserve enough structure that a later controller can consume:

* ranked passage lists,
* retrieval scores,
* rerank scores,
* provenance metadata,
* prompt context candidates,
* run-level config and manifest metadata.

---

# 13. Cluster execution requirements

## 13.1 Design principle

All main runs must be executable on the cluster through a non-interactive CLI.

## 13.2 Requirements

* no notebook-only execution paths
* all experiments launchable from config files
* support for batch execution over shards or dataset splits
* output directories must be unique, deterministic, and collision-safe
* failed runs should leave partial artifacts in an inspectable state
* support a tiny local config for quick debugging before cluster submission
* cluster paths and caches must be configurable
* run scripts should support environment bootstrapping through a clear entrypoint

## 13.3 Preferred requirements

* support job-array execution
* support resume-from-artifacts for long runs
* support per-shard metric merging
* support manifest files for multi-run sweeps

---

# 14. Testing requirements

## 14.1 Required test layers

### Unit tests

For:

* schema normalization
* prompt assembly
* parser behavior
* scoring functions
* config loading
* metadata hashing
* path resolution
* manifest generation

### Contract tests

For:

* `predictions.jsonl`
* `retrievals.jsonl`
* `prompts.jsonl`
* `evaluations.jsonl`
* `run_manifest.json`
* config and artifact completeness

These tests verify schema validity and required fields.

### Integration tests

For:

* end-to-end pipeline execution on a tiny fixture dataset,
* retrieval → rerank → prompt → generate → evaluate → log flow,
* adapter correctness,
* controller-ready interface compatibility.

### Regression tests

For:

* previously observed schema bugs,
* prompt formatting bugs,
* ordering bugs,
* logging omissions,
* retrieval / rerank metadata mismatches,
* run manifest omissions.

### Cluster smoke tests

For:

* CLI entrypoints,
* config parsing,
* output directory creation,
* environment variable handling,
* dry-run and tiny-run submission modes.

## 14.2 Test quality rules

* Tests should prefer deterministic fixtures.
* Tests should check outputs and contracts, not just process completion.
* Production code should not contain hidden test-only logic.
* All major modules should be introduced with tests.
* Every bug fix should introduce a regression test.

---

# 15. Recommended code modules

Suggested structure:

```text
rag_project/
  configs/
    local/
    cluster/
    tests/
  data/
  src/
    adapters/
      flashrag/
    data_loading/
    retrieval/
    reranking/
    prompting/
    generation/
    evaluation/
    logging/
    manifests/
    pipelines/
    cluster/
    utils/
  tests/
    unit/
    contracts/
    integration/
    regression/
    cluster_smoke/
  scripts/
    run_baseline.py
    run_local_smoke.py
    run_cluster_job.py
    evaluate_predictions.py
    inspect_examples.py
    validate_run_artifacts.py
  outputs/
```

This is intended to stay modular and explicit, closer to your existing style of separating schema, provider, prompt, and parsing responsibilities rather than hiding everything in one orchestration file. ([github.com](https://github.com/mohanwugupta/God-s-Reach/tree/main/designspace_extractor/llm))

## Module responsibilities

### `adapters/flashrag/`

Thin wrappers that translate between project schemas and FlashRAG objects.

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

### `manifests/`

Run manifests, provenance capture, config hashing, environment logging.

### `pipelines/`

End-to-end orchestration.

### `cluster/`

Cluster submission helpers, resource profiles, path resolution, shard handling.

---

# 16. Implementation tasks

## Task group A: testing scaffolding

* set up pytest structure
* add unit, contract, integration, regression, and cluster smoke test directories
* implement fixture dataset and tiny local smoke config
* add schema contract tests before pipeline implementation

## Task group B: framework integration

* pin FlashRAG version
* implement FlashRAG adapter wrappers
* test adapter conversions into project schemas
* create minimal baseline run through FlashRAG on a fixture dataset

## Task group C: dataset ingestion

* implement dataset loader
* normalize examples to common schema
* write dataset preview script

## Task group D: retrieval

* implement retriever wrapper
* return top-k passages with scores
* save retrieval outputs

## Task group E: reranking

* implement reranker wrapper
* rerank retrieval outputs
* save reranked lists

## Task group F: prompt construction

* implement fixed grounded-QA prompt template
* format retrieved passages deterministically

## Task group G: generation

* implement answer generation wrapper
* save raw outputs and parsed answer

## Task group H: evaluation

* implement normalized answer scoring
* generate summary metrics

## Task group I: orchestration

* implement config-driven run script
* save all artifacts under run directory
* capture run manifest

## Task group J: inspection

* implement script that samples successes and failures
* export small error pack for manual review

## Task group K: cluster execution

* implement cluster-ready CLI entrypoint
* implement cluster config profile
* implement run manifest capture with job metadata
* test output directory behavior under batch execution

## Task group L: regression hardening

* add regression tests for each bug found during implementation
* validate that saved artifacts are sufficient for downstream controller work

---

# 17. Exit criteria

PRD 1 is done when:

1. `run_baseline.py` completes end-to-end on at least one benchmark split through the FlashRAG-backed execution path,
2. outputs are saved in structured project-owned files,
3. summary metrics are computed automatically,
4. two rerunnable configs work without code changes,
5. one local smoke config and one cluster config both run successfully,
6. a qualitative pack of at least 25 examples is exported for inspection,
7. code paths are ready for later controller insertion,
8. tests cover core contracts and the end-to-end baseline path.

Preferred:

9. Baseline A and Baseline B have both been run on the same split.
10. the run manifest captures enough provenance to rerun the experiment later on the cluster.

---

# 18. Risks and mitigations

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

## Risk 5

Framework defaults or version changes silently alter behavior.

### Mitigation

Pin FlashRAG version, save full run manifests, and route all outputs through project-owned schemas. ([arXiv][1])

## Risk 6

Cluster execution introduces path, environment, or sharding bugs that do not appear locally.

### Mitigation

Require local smoke tests plus cluster smoke tests and explicit path/config handling.

## Risk 7

The code passes manual inspection but lacks stable behavior under iteration.

### Mitigation

Use TDD, regression tests for every bug, and contract tests for all saved artifacts.

---

# 19. Open decisions to settle now

1. Which FlashRAG-supported retriever for v1?
2. Will hybrid retrieval be default?
3. Which FlashRAG-supported reranker for v1?
4. Which generator model for v1?
5. What top-k and post-rerank k values will be default?
6. Which initial dataset or benchmark split will be used?
7. What confidence proxy, if any, should be logged now for future use?
8. What FlashRAG version or commit will be pinned?
9. What is the default cluster execution profile?
10. What local smoke-test fixture dataset will be used?
11. What minimum test coverage rules will be enforced before cluster runs?

---

# 20. Recommended default choices

For a pragmatic v1, I’d lean toward:

* **Framework:** FlashRAG pinned to a fixed commit or release
* **Execution:** local smoke test first, cluster as default for real runs
* **Testing:** pytest with unit + contract + integration + regression + cluster smoke layers
* **Run philosophy:** red → green → refactor for all major modules
* **Retriever:** hybrid if easy, otherwise strong dense retrieval
* **Reranker:** yes, one standard reranker
* **Generator:** one stable high-quality model with temperature 0
* **Top-k retrieval:** 8 to 10
* **Top-k after rerank:** 4 to 6
* **Prompt:** simple grounded QA prompt with instruction to answer from provided evidence only

The point is not squeezing maximum benchmark performance. The point is building a baseline people will respect.

---

# 21. Deliverables

At the end of PRD 1, you should have:

* baseline RAG pipeline code
* FlashRAG adapter layer
* run configs for local and cluster execution
* structured outputs
* run manifests
* automatic evaluation
* qualitative error pack
* test suite covering core contracts and smoke paths
* short README explaining how to rerun experiments


