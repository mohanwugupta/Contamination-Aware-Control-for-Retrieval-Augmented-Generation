# PRD 2 — Contamination Benchmark Suite

## Project name

**Contamination Benchmark Suite for Retrieval-Augmented Generation**

## Purpose

Build a benchmark suite and construction pipeline that isolates **contaminated retrieval** as a distinct RAG failure mode, while remaining fully compatible with the baseline systems and benchmark harness defined in PRD 1.

This stage is not about building a better model.
It is about building the data and evaluation substrate needed to show:

1. where standard RAG works,
2. where it fails due to contamination rather than pure absence of evidence,
3. how those failures differ across ambiguity, contradiction, misinformation, and missing-support settings,
4. and how later controller interventions should be evaluated.

The benchmark suite must be usable directly by the PRD 1 harness, including its support for:

* single-answer QA,
* multi-answer QA,
* unknown / abstain-compatible QA,
* dataset-specific evaluation adapters.

## Why this stage exists

PRD 1 gives the project a baseline matrix and benchmark harness.

PRD 2 must now provide a benchmark suite that makes the project’s central claim testable:

> some RAG failures happen not because evidence is absent, but because retrieved evidence is misleading in aggregate.

If PRD 2 is weak, later improvements will look like generic answer filtering, generic abstention, or generic robustness tricks rather than a targeted solution to contamination.

---

# 1. Core thesis of this stage

PRD 2 is not just “collect some hard examples.”

PRD 2 must build a **benchmark portfolio** that is aligned with PRD 1’s benchmark ladder:

* **Tier 1 core ambiguity/confusion**
* **Tier 2 context-faithfulness and answerability**
* **Tier 3 mixed ambiguity + misinformation + noise**
* plus a **controlled matched benchmark slice** that supports cleaner causal claims

This means PRD 2 has to do two jobs at once:

1. normalize and package natural benchmark slices from existing datasets,
2. construct project-owned controlled examples with matched clean / contaminated / missing variants.

That combination is what makes later controller evaluation credible.

---

# 2. Goals

## Primary goal

Construct a contamination benchmark suite with example-level conditions that distinguish:

* **clean retrieval**
* **contaminated retrieval**
* **missing-evidence retrieval**

for the same or closely matched questions, while remaining executable inside the PRD 1 baseline harness.

## Secondary goals

* create a reproducible contamination taxonomy,
* support both natural and controlled contamination settings,
* support both single-answer and multi-answer benchmark records,
* produce structured labels suitable for automatic evaluation and manual inspection,
* preserve enough provenance to reconstruct how every benchmark example was created,
* make the suite reusable beyond this specific project.

## Non-goals

This stage will **not**:

* implement the controller,
* optimize model performance,
* solve retrieval itself,
* create a giant all-purpose RAG benchmark covering every possible failure mode,
* let external dataset schemas define the project’s benchmark contract.

This benchmark should be narrow enough to stay interpretable and broad enough to support the paper’s main claims.

---

# 3. Benchmark ladder alignment with PRD 1

PRD 2 must align directly with the benchmark ladder already established in PRD 1.

## Tier 1 — Core project benchmark

### Dataset

**AmbigDocs** 

### Role in PRD 2

This is the primary natural benchmark for:

* same-name entity ambiguity,
* multi-answer behavior,
* merged-answer failures,
* incomplete disambiguation.

### Why it matters

AmbigDocs is the cleanest existing benchmark for the project’s ambiguity-centered failure mode, and its ontology of complete / partial / ambiguous / merged / no-answer outputs is directly relevant to PRD 1’s multi-answer evaluation requirements. 

## Tier 2 — Robustness / faithfulness benchmark

### Dataset

**FaithEval** 

### Role in PRD 2

This is the primary natural benchmark for:

* unanswerable contexts,
* inconsistent contexts,
* counterfactual contexts,
* distinguishing contamination from missing support.

### Why it matters

FaithEval explicitly separates unanswerable, inconsistent, and counterfactual contextual failures, which is crucial for preventing the project from collapsing contamination into generic failure-to-answer. 

## Tier 3 — Stretch / mixed-conflict benchmark

### Dataset

**RAMDocs** 

### Role in PRD 2

This is the primary natural stress test for:

* ambiguity,
* misinformation,
* noise,
* uneven support across possible answers.

### Why it matters

RAMDocs captures the hardest realistic mixed-conflict setting and is therefore the right stretch benchmark once the simpler contamination story is already working. 

## Supplementary natural analysis slices

### WikiContradict

Use for real-world contradictory evidence and contradiction-style contamination. 

### RAGTruth

Use for generation-side hallucination auditing relative to retrieved context. 

These are useful, but they are not the main ladder benchmarks. They support analysis, error typing, and future ablations.

## Controlled benchmark slice

In addition to the natural datasets above, PRD 2 must create a project-owned **controlled matched slice** built on top of a scorable factual QA source.

This is the most important benchmark component for causal interpretation.

---

# 4. Development philosophy

## 4.1 Test-driven development is required

This stage will use **test-driven development** as a default engineering method.

New benchmark functionality should be developed using a strict:

**red → green → refactor**

loop:

1. write a failing test for the intended benchmark behavior,
2. implement the smallest change needed to make the test pass,
3. refactor only once the relevant tests are green.

## 4.2 What must be protected by tests

The following must be protected by tests:

* condition definitions,
* contamination taxonomy mapping,
* benchmark schemas,
* dataset adapter correctness,
* split integrity,
* provenance capture,
* controlled construction logic,
* answer-mode packaging for PRD 1 compatibility,
* cluster execution entrypoints.

## 4.3 Engineering rules

* Do not weaken tests to accommodate ambiguous or sloppy labels.
* Every bug in construction, labeling, or splits must add a regression test.
* Prefer explicit schema checks over ad hoc inspection.
* Prefer adapters over changing the benchmark contract to fit external sources.
* Prefer small audited pilot slices before scaling up.

---

# 5. Execution environment

## 5.1 Cluster-first construction

Benchmark construction and validation should be designed to run primarily on the lab’s **cluster**.

The construction pipeline must therefore be:

* CLI-first,
* config-driven,
* shardable,
* batchable,
* resumable where possible,
* explicit about cache, output, and scratch paths,
* non-interactive by default.

## 5.2 Local execution still matters

The benchmark pipeline must also support:

* tiny local smoke-test builds,
* fixture-based adapter tests,
* quick small-sample inspection before cluster-scale runs.

Cluster is the default for serious construction and validation.
Local is required for development safety.

---

# 6. Core benchmark thesis

A retrieved set should not be judged only by whether it contains relevant material.

Instead, the benchmark should test whether the retrieved set is:

* **supportive and coherent**
* **misleading in aggregate**
* **insufficient for answering**

The central distinction is:

> some retrieved sets contain relevant evidence but still create failure because the set as a whole is unstable, conflicting, ambiguity-inducing, or synthesis-inducing.

The benchmark must make that distinction visible both:

* in natural datasets,
* and in matched controlled examples.

---

# 7. Success criteria

PRD 2 is complete if all of the following are true:

1. the suite contains clearly labeled **clean**, **contaminated**, and **missing-evidence** conditions,
2. contamination types are defined in a reproducible taxonomy,
3. the suite can be executed through the PRD 1 baseline harness without schema hacks,
4. the benchmark supports PRD 1 answer modes:

   * `single`
   * `multi`
   * `unknown_or_abstain`
5. at least one baseline RAG system performs measurably worse on contaminated than clean retrieval,
6. contaminated examples can be manually inspected and understood,
7. benchmark construction is scriptable and reproducible,
8. there is a dev/tuning split and a held-out eval split,
9. major components are covered by tests under a red → green → refactor workflow.

Preferred:

10. the benchmark includes both naturally occurring and synthetically induced contamination,
11. both local smoke builds and cluster-scale builds succeed,
12. Tier 1, Tier 2, and Tier 3 natural slices are all integrated.

---

# 8. Benchmark design principles

## Principle 1: contamination must be answer-relevant

A contaminated passage is not just irrelevant.

It must be plausible enough to influence synthesis or answer choice.

Bad contamination:

* a random sports paragraph in a finance question

Good contamination:

* a paragraph about a similarly named company with a conflicting acquisition date

## Principle 2: contamination must be separable from missing evidence

The benchmark must avoid collapsing these into one category.

### Missing-evidence condition

The retrieved set does not provide enough support.

### Contaminated condition

The retrieved set includes evidence that appears relevant but pushes toward the wrong answer, merged answer, unstable answer, or false synthesis.

## Principle 3: contamination should come in interpretable subtypes

This helps later controller design and ablations.

## Principle 4: the benchmark must support case-level analysis

Each example should be inspectable by a human with:

* query
* gold answer(s)
* retrieved passages
* labels
* rationale
* provenance

## Principle 5: benchmark semantics are project-owned

External datasets do not define:

* final condition labels,
* final contamination subtype labels,
* final answerability labels,
* final packaging into PRD 1 answer modes.

The project owns those decisions.

---

# 9. Condition definitions

## 9.1 Clean retrieval

### Definition

The retrieved set contains sufficient, mutually supportive evidence for the correct answer.

### Requirements

* answerable from retrieved context
* no major answer-critical contradictions
* no severe entity ambiguity
* no obvious lure-like support for wrong alternatives

## 9.2 Contaminated retrieval

### Definition

The retrieved set contains enough relevant-looking or semantically overlapping evidence to appear useful, but the set as a whole is misleading, internally unstable, ambiguity-inducing, or supportive of competing answer hypotheses.

### Requirements

* at least one plausible wrong-support or destabilizing passage is present
* the contamination is answer-relevant
* the example remains distinguishable from pure missing-support failure

## 9.3 Missing-evidence retrieval

### Definition

The retrieved set lacks enough support for the question, regardless of whether some superficially related material is present.

### Requirements

* no passage subset provides reliable support for the gold answer
* failure arises primarily from insufficiency rather than conflict

## 9.4 Harmless-noise optional label

### Definition

The retrieved set includes weakly relevant or irrelevant material, but it should not materially change the answer if the model is behaving well.

This label is useful because it helps distinguish true contamination from generic clutter.

---

# 10. Contamination taxonomy

This taxonomy should be fixed early and used throughout.

## Required contamination types

### T1. Conflicting answer-bearing detail

Two or more passages support different values for the answer-critical field.

Examples:

* conflicting dates
* conflicting numbers
* conflicting entities
* conflicting status claims

### T2. Same-name / entity ambiguity

Passages refer to different entities sharing the same or similar surface form.

Examples:

* people with the same name
* organizations with related names
* recurring event titles in different years

### T3. Partial-match lure

Passages match the query topic or framing but differ on the crucial answer-bearing attribute.

Examples:

* right company, wrong acquisition event
* right person, wrong role
* right event family, wrong edition/year

### T4. Unsupported synthesis

No single coherent subset supports the generated wrong answer, but combining fragments from multiple passages makes it seem plausible.

### T5. Retrieval instability

The apparent answer support changes substantially under:

* passage reorderings,
* leave-one-out removal,
* top-k changes,
* near-duplicate substitutions.

This can either be a label or an induced evaluation condition.

## Optional contamination types

### T6. Temporal mismatch

Passages describe different time slices that are each plausible but outdated or inconsistent relative to the query.

### T7. Near-duplicate misleading reinforcement

Several passages repeat the same wrong or misleading cue, crowding out correct support.

---

# 11. Natural benchmark portfolio

PRD 2 should mix **natural** and **controlled** data.

## 11.1 Tier 1 natural slice — AmbigDocs

Use for:

* same-name ambiguity,
* multi-answer generation,
* merged-answer failures,
* explicit disambiguation requirements. 

Role:

* primary natural benchmark for ambiguity-centered contamination.

## 11.2 Tier 2 natural slice — FaithEval

Use for:

* inconsistent context,
* unanswerable context,
* context-faithfulness failure,
* contamination vs missing-support distinction. 

Role:

* primary natural benchmark for distinguishing inconsistency from insufficiency.

## 11.3 Tier 3 natural slice — RAMDocs

Use for:

* ambiguity,
* misinformation,
* noise,
* mixed conflict in one retrieved set. 

Role:

* main stretch benchmark for realistic mixed-conflict evaluation.

## 11.4 Supplementary contradiction slice — WikiContradict

Use for:

* contradiction-style contamination,
* explicit vs implicit conflict,
* real-world conflicting evidence from a trusted source. 

## 11.5 Supplementary hallucination-audit slice — RAGTruth

Use for:

* model-output hallucination relative to retrieved context,
* span-level or example-level conflict auditing,
* downstream analysis of whether contamination correlates with hallucination type. 

---

# 12. Controlled benchmark slice

The controlled slice is likely the most important component for the research story.

Construct project-owned contamination conditions on top of a scorable factual QA source compatible with PRD 1’s single-answer pipeline, such as an NQ-like seed source.

## Purpose

* isolate causal effects of contamination,
* support within-item comparison across conditions,
* let PRD 1 baselines be compared on matched clean / contaminated / missing variants.

## Controlled format

For each base question, aim to create 3 aligned variants:

### Variant A — Clean

Supportive evidence only.

### Variant B — Contaminated

Start from the clean set and inject contamination.

### Variant C — Missing evidence

Remove or replace answer-supporting passages so the question becomes unanswerable from the retrieved set.

This matched structure makes the later paper much stronger.

---

# 13. Controlled benchmark record format

Each controlled example should package cleanly into PRD 1’s harness.

## Example structure

```json id="xqv5ph"
{
  "example_id": "q_104",
  "question": "When did Company Y acquire Company X?",
  "task_type": "single_answer_qa",
  "gold": {
    "single_answer": "2019",
    "multi_answers": null,
    "unknown_allowed": true
  },
  "condition_sets": {
    "clean": {
      "passages": ["..."],
      "labels": {
        "answerable": true,
        "condition": "clean"
      }
    },
    "contaminated": {
      "passages": ["..."],
      "labels": {
        "answerable": true,
        "condition": "contaminated",
        "contamination_types": ["T1", "T3"]
      }
    },
    "missing": {
      "passages": ["..."],
      "labels": {
        "answerable": false,
        "condition": "missing_evidence"
      }
    }
  }
}
```

---

# 14. Benchmark record schema

Each benchmark item should include enough information to support both analysis and PRD 1 execution.

```json id="l9zu3i"
{
  "example_id": "ex_001",
  "question": "Who founded Organization Z?",
  "task_type": "single_answer_qa",
  "gold": {
    "single_answer": "Alice Smith",
    "multi_answers": null,
    "unknown_allowed": false
  },
  "dataset_source": "controlled_v1",
  "condition": "contaminated",
  "answerable_from_context": true,
  "answer_mode_for_prd1": "single",
  "passages": [
    {
      "passage_id": "p1",
      "text": "...",
      "source_doc": "doc_14",
      "role_label": "correct_support"
    },
    {
      "passage_id": "p2",
      "text": "...",
      "source_doc": "doc_88",
      "role_label": "entity_lure"
    }
  ],
  "labels": {
    "contamination_types": ["T2"],
    "harmless_noise_present": false,
    "rationale": "p2 refers to a different organization with a similar name and conflicting founder information"
  },
  "provenance": {
    "builder": "controlled_injection_v1",
    "source_dataset": "base_qa_source",
    "source_example_id": "orig_001",
    "construction_recipe_id": "recipe_17"
  }
}
```

## Required provenance fields

Every benchmark item must preserve:

* source dataset
* source example ID
* adapter or builder version
* recipe ID where applicable
* benchmark version
* split assignment source

---

# 15. PRD 1 compatibility requirements

PRD 2 must be consumable by the PRD 1 harness.

## Required compatibility

Each benchmark example must cleanly map into one of PRD 1’s answer modes:

* `single`
* `multi`
* `unknown_or_abstain`

## Required adapters

PRD 2 must provide dataset adapters that convert benchmark records into PRD 1 normalized example schema while preserving:

* example ID
* passage order
* provenance
* answer mode
* condition label
* contamination subtype metadata

## Adapter rule

If an external dataset does not match the project schema, write an adapter.
Do not weaken the project schema to match the raw external dataset.

---

# 16. Construction methods

## Method A: natural dataset normalization

For each of:

* AmbigDocs
* FaithEval
* RAMDocs
* WikiContradict
* RAGTruth

implement a dataset adapter that:

* loads raw examples,
* maps them into project benchmark schema,
* attaches condition labels,
* attaches contamination labels,
* stores provenance metadata.

## Method B: retrieval replay

Take a base query set and run the PRD 1 baseline retriever. Then identify:

* supportive passages,
* contamination candidates,
* missing-support variants.

## Method C: contaminant injection

Start from a clean set and add one or more passages designed to create:

* contradiction,
* ambiguity,
* partial-match lure,
* temporal mismatch.

## Method D: support removal

Create missing-evidence variants by removing core support passages while keeping the set plausible.

## Method E: perturbation generation

Create alternate retrieval presentations by:

* shuffling passage order,
* leaving out one passage,
* swapping near neighbors,
* varying k.

This is especially useful for instability labels and later controller stress tests.

---

# 17. Labeling requirements

## Required labels per example

* condition: `clean`, `contaminated`, or `missing_evidence`
* answerable_from_context: `true/false`
* contamination_types: list
* rationale: short human-readable explanation
* source dataset name
* passage role labels
* answer mode for PRD 1 compatibility

## Required passage role labels

Each passage should be labeled with one of:

* `correct_support`
* `conflicting_support`
* `entity_lure`
* `partial_match_lure`
* `harmless_noise`
* `insufficient_context`
* `duplicate_or_reinforcing`
* `other`

## Labeling quality control

At least a subset of examples should be manually audited.

Recommended:

* dual annotation on an initial sample
* resolve disagreements
* refine taxonomy before scaling

Special care should be taken with:

* FaithEval, where inconsistent vs unanswerable must remain separate,
* AmbigDocs, where ambiguity labels and merged-answer implications must remain precise,
* RAMDocs, where ambiguity, misinformation, and noise must not be collapsed,
* RAGTruth, where generation-side hallucination labels should not be treated as retrieval-side contamination without inspection.

---

# 18. Split requirements

The benchmark should have at least:

* **pilot**
* **dev/tuning**
* **held-out eval**

## Split design rule

Do not let near-duplicate base questions leak across dev and eval splits.

Avoid leakage across:

* natural and controlled variants of essentially the same example,
* multiple condition versions of the same controlled item,
* recipe variants of the same base question.

## Split integrity requirement

Split assignment must be reproducible from frozen metadata and test-covered.

---

# 19. Size requirements

For the MVP, the suite does not need to be huge.

A strong starting target is:

* **100 to 300 high-quality matched controlled items**
* plus modest but real natural slices from:

  * AmbigDocs
  * FaithEval
  * RAMDocs

Supplementary slices from WikiContradict and RAGTruth can start smaller.

Quality matters more than scale.

---

# 20. Benchmark validation requirements

This PRD is not just about building data.
It must show the data is useful.

## Required benchmark validation checks

### Check 1

Baseline RAG accuracy is high on clean variants.

### Check 2

Baseline RAG degrades on contaminated variants.

### Check 3

Missing-evidence variants show a different failure profile from contaminated variants.

### Check 4

Manual inspection confirms contaminated examples are genuinely misleading rather than simply irrelevant.

### Check 5

Contamination subtypes can be recognized by humans from labels and rationales.

### Check 6

The benchmark can be run through PRD 1 baseline variants without adapter breakage.

### Check 7

Meaningful performance differences are visible on at least some of:

* AmbigDocs
* FaithEval
* RAMDocs

Preferred:

* WikiContradict
* RAGTruth

---

# 21. Testing requirements

## 21.1 Required test layers

### Unit tests

For:

* schema validation,
* label validation,
* taxonomy mapping,
* provenance generation,
* config loading,
* path resolution.

### Contract tests

For:

* benchmark JSONL/parquet schema,
* split manifests,
* label files,
* benchmark manifest completeness,
* PRD 1 adapter compatibility.

### Integration tests

For:

* natural dataset adapter output,
* controlled construction from base item to aligned variants,
* validation pipeline execution,
* compatibility with PRD 1 input expectations.

### Regression tests

For:

* prior labeling bugs,
* schema drift,
* split leakage bugs,
* provenance omissions,
* adapter ID mismatches,
* controlled construction bugs.

### Cluster smoke tests

For:

* benchmark build CLI entrypoints,
* config parsing,
* shard-aware construction,
* output directory creation,
* validation job startup.

## 21.2 Test quality rules

* Prefer deterministic fixtures and tiny sample datasets.
* Test semantic outputs, not just process completion.
* Do not hide benchmark logic in test-only branches.
* Every benchmark bug fix should add a regression test.

---

# 22. Required outputs

## Data artifacts

* normalized benchmark JSONL or parquet files
* condition-aligned example groups
* label files
* split files
* provenance metadata
* benchmark manifest

## Scripts

* benchmark construction script
* natural dataset adapter scripts
* contaminant injection script
* validation script
* inspection script
* artifact validation script

## Reports

* contamination taxonomy doc
* benchmark summary stats
* pilot validation memo
* sampled qualitative examples
* dataset coverage summary for:

  * AmbigDocs
  * FaithEval
  * RAMDocs
  * WikiContradict
  * RAGTruth

---

# 23. Suggested code organization

```text
rag_project/
  benchmark/
    raw/
    processed/
    splits/
    metadata/
    manifests/
  configs/
    benchmark/
      local/
      cluster/
      tests/
  src/
    benchmark_construction/
    natural_adapters/
    contamination_injection/
    labeling/
    validation/
    inspection/
    manifests/
    cluster/
    utils/
  tests/
    unit/
    contracts/
    integration/
    regression/
    cluster_smoke/
  scripts/
    build_benchmark.py
    validate_benchmark.py
    inspect_benchmark.py
    validate_benchmark_artifacts.py
    run_benchmark_local_smoke.py
    run_benchmark_cluster_job.py
```

---

# 24. Implementation tasks

## Task group A: testing scaffolding

* set up pytest structure
* add unit, contract, integration, regression, and cluster smoke directories
* implement tiny fixture datasets
* add schema contract tests before benchmark logic

## Task group B: taxonomy finalization

* finalize contamination subtype list
* finalize condition definitions
* write labeling guide
* write benchmark schema guide

## Task group C: natural dataset adapters

* implement AmbigDocs adapter
* implement FaithEval adapter
* implement RAMDocs adapter
* implement WikiContradict adapter
* implement RAGTruth adapter
* test adapter outputs

## Task group D: controlled source selection

* choose base QA source for controlled slice
* define inclusion / exclusion criteria
* define answerability rules
* define mapping into PRD 1 answer modes

## Task group E: controlled construction

* implement clean-set assembly
* implement contamination injection
* implement missing-support generation
* attach provenance metadata

## Task group F: labeling

* assign example-level and passage-level labels
* create rationales
* audit a subset manually
* record labeling guide version

## Task group G: split generation

* generate pilot, dev, and eval splits
* test for leakage across related items
* freeze split manifests

## Task group H: validation

* run PRD 1 baseline RAG on benchmark variants
* verify clean vs contaminated vs missing separation
* inspect performance separately on AmbigDocs, FaithEval, and RAMDocs
* validate adapter stability into PRD 1 format

## Task group I: packaging

* export benchmark files
* export split definitions
* produce summary statistics
* write benchmark manifest

## Task group J: cluster execution

* implement cluster-ready build and validation entrypoints
* implement cluster config profiles
* test shard-safe construction and artifact writing

## Task group K: regression hardening

* add regression tests for each bug found during development
* validate that frozen benchmark artifacts are sufficient for PRD 3 and PRD 4

---

# 25. Exit criteria

PRD 2 is done when:

1. a benchmark artifact exists with labeled clean, contaminated, and missing-evidence conditions,
2. contamination subtypes are documented and applied,
3. the PRD 1 baseline stack runs end-to-end on the benchmark suite,
4. the benchmark supports PRD 1 answer modes cleanly,
5. the benchmark shows measurable degradation under contaminated conditions,
6. a human-readable inspection pack exists with representative examples,
7. dev and eval splits are frozen,
8. benchmark artifacts are versioned, validated, and reproducible,
9. tests cover core schemas, adapters, split integrity, and validation paths.

Preferred:

10. AmbigDocs, FaithEval, and RAMDocs are all integrated in normalized form.
11. WikiContradict and RAGTruth are integrated as supplementary slices.
12. one local smoke build and one cluster build both succeed.

---

# 26. Risks and mitigations

## Risk 1

Contamination examples end up being just generic irrelevant context.

### Mitigation

Require answer-relevant contamination and rationale labels.

## Risk 2

Contaminated and missing-evidence conditions blur together.

### Mitigation

Leverage FaithEval explicitly for inconsistent-context vs unanswerable distinctions and require explicit `answerable_from_context` labeling. 

## Risk 3

Synthetic contamination looks unnatural.

### Mitigation

Mix controlled construction with natural slices from AmbigDocs, FaithEval, and RAMDocs, with supplementary contradiction and hallucination slices from WikiContradict and RAGTruth.   

## Risk 4

The taxonomy is too broad and inconsistent.

### Mitigation

Start with 4 to 5 core types only and map natural datasets onto them.

## Risk 5

Benchmark construction becomes too expensive.

### Mitigation

Start with a small, high-quality pilot benchmark rather than aiming for scale.

## Risk 6

Adapter layers silently alter benchmark meaning.

### Mitigation

Use contract tests, provenance checks, and PRD 1 compatibility tests.

## Risk 7

Cluster execution introduces split, path, or shard bugs that do not appear locally.

### Mitigation

Require local smoke tests plus cluster smoke tests and explicit path/config handling.

---

# 27. Recommended MVP version

The smallest strong version of PRD 2 is:

* one controlled base factual QA source
* natural slices drawn from:

  * AmbigDocs
  * FaithEval
  * RAMDocs
* a matched clean / contaminated / missing setup
* 4 core contamination types
* 100 to 200 pilot controlled examples
* manual inspection pack
* baseline validation results
* frozen benchmark manifest
* tested local and cluster build paths

Supplementary:

* small WikiContradict slice
* small RAGTruth slice

That is enough to make the failure mode concrete and align cleanly with PRD 1.

---

# 28. Open decisions to settle next

These need to be fixed before implementation:

1. how many examples to pull from each of:

   * AmbigDocs
   * FaithEval
   * RAMDocs
   * WikiContradict
   * RAGTruth
2. which of these are pilot-only vs main-benchmark datasets
3. whether initial contamination injection is manual, heuristic, or LLM-assisted
4. how much manual labeling budget to spend on pilot examples
5. whether instability is treated as a contamination subtype or a separate evaluation condition
6. what benchmark versioning scheme will be used
7. what local smoke-test fixture will be used
8. what minimum test coverage rules must be satisfied before cluster-scale builds

---

# 29. Acceptance criteria

This PRD has succeeded if, after running PRD 1 systems on it, you can credibly say:

* “Here are clean cases where standard RAG works.”
* “Here are contaminated cases where standard RAG fails despite relevant support being present.”
* “Here are missing-evidence cases that fail for a different reason.”
* “These categories are consistent enough to drive controller design.”
* “These patterns appear on the benchmarks PRD 1 actually uses: AmbigDocs, FaithEval, and RAMDocs.”
* “The benchmark suite is versioned, auditable, and stable enough for later paper evaluation.”

That is the real purpose of this stage.
