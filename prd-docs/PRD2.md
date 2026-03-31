# PRD 2 — Contamination Benchmark Suite

## Project name

**Contamination Benchmark Suite for Retrieval-Augmented Generation**

## Purpose

Build a benchmark and evaluation protocol that isolates **contaminated retrieval** as a distinct RAG failure mode, separate from:

* clean supportive retrieval
* missing-evidence retrieval
* harmless noisy retrieval

This benchmark suite will be used to:

1. diagnose whether contamination is a real and measurable issue in baseline RAG systems,
2. support development of the contamination-aware controller,
3. provide the core experimental testbed for later paper results.

## Why this stage exists

The project’s central claim depends on contamination being:

* **operationally defined**
* **present in realistic settings**
* **distinguishable from missing support**
* **controllable enough for experimentation**

If this stage is weak, later gains will look like generic abstention or filtering rather than a solution to a specific failure mode.

---

# 1. Goals

## Primary goal

Construct a benchmark suite with example-level conditions that distinguish:

* clean retrieval
* contaminated retrieval
* missing-evidence retrieval

for the same or closely matched questions.

## Secondary goals

* create a taxonomy of contamination types,
* support both natural and controlled contamination settings,
* produce structured labels suitable for automatic evaluation and manual inspection,
* make the suite reusable beyond this specific project.

## Non-goals

This stage will **not**:

* implement the controller,
* optimize model performance,
* solve retrieval itself,
* create a giant benchmark covering every possible RAG failure mode.

This benchmark should be narrow and targeted.

---

# 2. Core benchmark thesis

A retrieved set should not be judged only by whether it contains relevant material.

Instead, the benchmark should test whether the retrieved set is:

* **supportive and coherent**
* **misleading in aggregate**
* **insufficient for answering**

The key research distinction is:

> Some RAG failures happen even when relevant evidence is present, because the retrieved set contains semantically misleading mixtures that induce false synthesis.

The benchmark must make that claim testable.

---

# 3. Success criteria

PRD 2 is complete if all of the following are true:

1. the suite contains clearly labeled clean, contaminated, and missing-evidence conditions,
2. contamination types are defined in a reproducible taxonomy,
3. at least one baseline RAG system performs measurably worse on contaminated than clean retrieval,
4. contaminated examples can be manually inspected and understood,
5. benchmark construction is scriptable and reproducible,
6. there is a dev/tuning split and a held-out eval split.

Preferred:
7. the benchmark includes both naturally occurring and synthetically induced contamination.

---

# 4. Benchmark design principles

## Principle 1: contamination must be answer-relevant

A contaminated passage is not just irrelevant. It must be plausible enough to influence synthesis.

Bad contamination example:

* a random irrelevant sports paragraph in a finance question

Good contamination example:

* a paragraph about a similarly named company with a conflicting acquisition date

## Principle 2: contamination must be separable from missing evidence

The benchmark must avoid collapsing these into one category.

### Missing-evidence condition

The retrieved set does not provide enough support.

### Contaminated condition

The retrieved set includes evidence that appears relevant but pushes toward the wrong answer or destabilizes support.

## Principle 3: contamination should come in interpretable subtypes

This helps later controller design and ablations.

## Principle 4: the benchmark must support case-level analysis

Each example should be inspectable by a human with:

* query
* gold answer
* retrieved passages
* labels
* rationale

---

# 5. Condition definitions

## 5.1 Clean retrieval

Definition:
The retrieved set contains sufficient, mutually supportive evidence for the correct answer.

Requirements:

* answerable from retrieved context
* no major answer-critical contradictions
* no severe entity ambiguity
* no obvious lure-like support for wrong alternatives

## 5.2 Contaminated retrieval

Definition:
The retrieved set contains enough relevant-looking or semantically overlapping evidence to appear useful, but the set as a whole is misleading, internally unstable, or supports competing answer hypotheses.

Requirements:

* at least one plausible wrong-support or destabilizing passage is present
* the contamination is answer-relevant
* the example remains distinguishable from pure missing-support failure

## 5.3 Missing-evidence retrieval

Definition:
The retrieved set lacks enough support for the question, regardless of whether some superficially related material is present.

Requirements:

* no passage subset provides reliable support for the gold answer
* failure arises primarily from insufficiency rather than conflict

## 5.4 Harmless-noise optional label

Definition:
The retrieved set includes weakly relevant or irrelevant material, but it should not materially change the answer if the model is behaving well.

This label is useful because it helps distinguish true contamination from generic context clutter.

---

# 6. Contamination taxonomy

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

This is a high-value contamination type for the project.

### T5. Retrieval instability

The apparent answer support changes substantially under:

* passage reorderings
* leave-one-out removal
* top-k changes
* near-duplicate substitutions

This can either be a label or an induced evaluation condition.

## Optional contamination types

### T6. Temporal mismatch

Passages describe different time slices that are each plausible but outdated or inconsistent relative to the query.

### T7. Near-duplicate misleading reinforcement

Several passages repeat the same wrong or irrelevant but plausible cue, crowding out the correct support.

---

# 7. Benchmark construction strategy

The suite should mix **natural** and **controlled** data.

## 7.1 Natural contamination datasets

These datasets should be explicitly included as the natural benchmark slice because they cover different aspects of contamination and help ground the project in already observed RAG failure modes.

### WikiContradict

Use for:

* conflicting retrieved passages
* answer-bearing contradiction
* disagreement across evidence that appears relevant

Role in benchmark:

* primary source for contradiction-style contamination
* useful for T1 conflicting answer-bearing detail

### FaithEval

Use for:

* inconsistent context
* answerability distinctions
* clean separation between contaminated retrieval and unanswerable / missing-support conditions

Role in benchmark:

* primary source for distinguishing contaminated context from missing evidence
* especially useful for validating the project’s core conceptual distinction

### AmbigDocs

Use for:

* same-name entity ambiguity
* ambiguity-driven retrieval contamination
* confusion between superficially matching but distinct entities

Role in benchmark:

* primary source for T2 same-name / entity ambiguity
* useful for showing that contamination can come from lexical match without true semantic support

### RAGTruth

Use for:

* hallucination auditing
* conflict-with-context labels
* identifying cases where model outputs are unsupported or contradicted by retrieved evidence

Role in benchmark:

* primary source for auditing generation failures relative to context
* useful for downstream error analysis and evaluating whether contamination correlates with hallucination types

### RAMDocs

Use for:

* mixed ambiguity
* misinformation
* noisy or misleading retrieved context
* realistic combinations of contamination-like failure modes

Role in benchmark:

* broad natural stress test for mixed contamination
* especially useful for blended cases that involve ambiguity, distractors, and misleading support at once

## 7.2 Role of natural datasets in the suite

These natural datasets should not just be listed. They should be assigned concrete roles:

* **WikiContradict** → contradiction-focused natural slice
* **FaithEval** → inconsistent-context vs unanswerable distinction slice
* **AmbigDocs** → entity ambiguity slice
* **RAGTruth** → hallucination audit and context-conflict analysis slice
* **RAMDocs** → mixed contamination stress-test slice

Together, they provide a natural benchmark portfolio covering:

* contradiction
* ambiguity
* misinformation
* unsupported generation
* contamination vs missing-support separation

## 7.3 Controlled benchmark slice

Construct your own contamination conditions on top of a base QA example set.

Purpose:

* isolate causal effects of contamination
* support within-item comparison across conditions

This is probably the most important component for the research story.

---

# 8. Controlled benchmark format

For each base question, aim to create 3 aligned variants:

### Variant A — Clean

Supportive evidence only.

### Variant B — Contaminated

Start from the clean set and inject contamination.

### Variant C — Missing evidence

Remove or replace answer-supporting passages so the question becomes unanswerable from the retrieved set.

This makes later claims much stronger because performance differences can be compared within matched items.

## Example structure

```json id="0m9u2w"
{
  "example_id": "q_104",
  "question": "When did Company Y acquire Company X?",
  "gold_answer": "2019",
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

# 9. Benchmark record schema

Each benchmark item should include:

```json id="6oqgzw"
{
  "example_id": "ex_001",
  "question": "Who founded Organization Z?",
  "gold_answer": "Alice Smith",
  "dataset_source": "controlled_v1",
  "condition": "contaminated",
  "answerable_from_context": true,
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
  }
}
```

---

# 10. Data sources and acquisition requirements

## Natural-source requirements

The natural slice should be built explicitly from:

* **WikiContradict**
* **FaithEval**
* **AmbigDocs**
* **RAGTruth**
* **RAMDocs**

These should be normalized into a common schema, with each example mapped where possible to:

* condition label
* contamination subtype
* answerability label
* passage role labels
* short rationale

## Controlled-source requirements

For the controlled slice, choose a base set of QA examples where:

* the answer is objectively scorable,
* supporting context can be curated or retrieved,
* contamination can be injected without making the example nonsensical.

## Quality rule

Do not build the benchmark from vague open-ended questions where correctness is hard to score.

Favor questions with:

* short factual answers
* answer-critical fields
* clear textual evidence

---

# 11. Construction methods

## Method A: natural dataset normalization

For each of:

* WikiContradict
* FaithEval
* AmbigDocs
* RAGTruth
* RAMDocs

implement a dataset adapter that:

* loads raw examples
* maps them into the benchmark schema
* attaches contamination labels
* stores provenance metadata

## Method B: retrieval replay

Take a base query set and run the baseline retriever. Then manually or semi-automatically identify:

* supportive passages
* contamination candidates
* missing-support variants

## Method C: contaminant injection

Start from a clean set and add one or more passages designed to create:

* answer conflict
* entity ambiguity
* partial-match lure
* temporal mismatch

## Method D: support removal

Create missing-evidence variants by removing the core support passages while keeping surrounding context plausible.

## Method E: perturbation generation

Create alternate retrieval presentations by:

* shuffling passage order
* leaving out one passage
* swapping near neighbors
* varying k

This is especially useful for instability labels.

---

# 12. Labeling requirements

## Required labels per example

* condition: `clean`, `contaminated`, or `missing_evidence`
* answerable_from_context: `true/false`
* contamination_types: list
* rationale: short human-readable explanation
* source dataset name
* passage role labels

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

Special care should be taken to audit examples imported from:

* FaithEval, where inconsistent context vs unanswerable distinctions matter
* AmbigDocs, where ambiguity labels need to remain precise
* RAGTruth, where generation-side hallucination labels should not be conflated with retrieval-side contamination without inspection

---

# 13. Split requirements

The benchmark should have at least:

* **dev/tuning split**
* **held-out eval split**

If possible:

* **pilot slice** for quick iteration
* **main eval slice** for reported results

## Split design rule

Do not let near-duplicate base questions leak across dev and eval splits.

Also avoid leakage across natural and controlled variants of essentially the same example.

---

# 14. Size requirements

For the MVP, the suite does not need to be huge.

A good starting target is:

* 100 to 300 high-quality matched items for pilot work
* each with clean, contaminated, and missing variants where possible

For the natural slice, aim for coverage across all five named datasets, even if initial counts are modest.

Preferred composition:

* contradiction examples from WikiContradict
* inconsistent-context / unanswerable examples from FaithEval
* ambiguity examples from AmbigDocs
* hallucination-audit examples from RAGTruth
* mixed contamination examples from RAMDocs

Quality matters more than scale at this stage.

---

# 15. Evaluation requirements for the benchmark itself

This PRD is not just about building data. It must show the data is useful.

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

Contamination subtypes can be recognized by humans from the labels and rationales.

### Check 6

Performance differences are visible on at least some of the named natural datasets:

* WikiContradict
* FaithEval
* AmbigDocs
* RAGTruth
* RAMDocs

This matters because the benchmark should show contamination is not just a synthetic artifact.

---

# 16. Required outputs

## Data artifacts

* normalized benchmark JSONL or parquet files
* condition-aligned example groups
* label files
* split files
* provenance metadata

## Scripts

* benchmark construction script
* natural dataset adapter scripts
* contaminant injection script
* validation script
* inspection script

## Reports

* contamination taxonomy doc
* benchmark summary stats
* pilot validation memo
* sampled qualitative examples
* dataset coverage summary for:

  * WikiContradict
  * FaithEval
  * AmbigDocs
  * RAGTruth
  * RAMDocs

---

# 17. Suggested code organization

```text
rag_project/
  benchmark/
    raw/
    processed/
    splits/
    metadata/
  src/
    benchmark_construction/
    natural_adapters/
    contamination_injection/
    labeling/
    validation/
    inspection/
  scripts/
    build_benchmark.py
    validate_benchmark.py
    inspect_benchmark.py
```

---

# 18. Implementation tasks

## Task group A: taxonomy finalization

* finalize contamination subtype list
* finalize condition definitions
* write labeling guide

## Task group B: natural dataset adapters

* implement WikiContradict adapter
* implement FaithEval adapter
* implement AmbigDocs adapter
* implement RAGTruth adapter
* implement RAMDocs adapter
* map each into common schema

## Task group C: source selection

* choose base QA sources for controlled variants
* define inclusion/exclusion criteria
* define which natural datasets are used in pilot vs full benchmark

## Task group D: controlled construction

* implement clean-set assembly
* implement contamination injection
* implement missing-support generation

## Task group E: labeling

* assign example-level and passage-level labels
* create rationales
* audit a subset manually

## Task group F: validation

* run baseline RAG on benchmark variants
* verify clean vs contaminated vs missing separation
* inspect performance separately on the five natural datasets

## Task group G: packaging

* export benchmark files
* export split definitions
* produce summary statistics

---

# 19. Exit criteria

PRD 2 is done when:

1. a benchmark file exists with labeled clean, contaminated, and missing-evidence conditions,
2. contamination subtypes are documented and applied,
3. the baseline RAG stack from PRD 1 runs end-to-end on the benchmark,
4. the benchmark shows measurable degradation under contaminated conditions,
5. a human-readable inspection pack exists with representative examples,
6. dev and eval splits are frozen.

Preferred:
7. all five natural contamination datasets are integrated in normalized form:

* WikiContradict
* FaithEval
* AmbigDocs
* RAGTruth
* RAMDocs

---

# 20. Risks and mitigations

## Risk 1

Contamination examples end up being just generic irrelevant context.

### Mitigation

Require answer-relevant contamination and rationale labels.

## Risk 2

Contaminated and missing-evidence conditions blur together.

### Mitigation

Leverage FaithEval explicitly for inconsistent-context vs unanswerable distinctions, and require explicit `answerable_from_context` labeling.

## Risk 3

Synthetic contamination looks unnatural.

### Mitigation

Mix controlled construction with natural datasets including WikiContradict, AmbigDocs, RAGTruth, and RAMDocs.

## Risk 4

The taxonomy is too broad and inconsistent.

### Mitigation

Start with 4 to 5 core types only and map the named datasets onto them.

## Risk 5

Benchmark construction becomes too expensive.

### Mitigation

Start with a small, high-quality pilot benchmark rather than aiming for scale.

---

# 21. Recommended MVP version

The smallest strong version of PRD 2 is:

* one base factual QA source
* natural slices drawn from:

  * WikiContradict
  * FaithEval
  * AmbigDocs
  * RAGTruth
  * RAMDocs
* a matched clean / contaminated / missing setup
* 4 core contamination types
* 100 to 200 pilot examples
* manual inspection pack
* baseline validation results

That is enough to make the failure mode concrete.

---

# 22. Open decisions to settle next

These need to be fixed before implementation:

1. how many examples to pull from each of:

   * WikiContradict
   * FaithEval
   * AmbigDocs
   * RAGTruth
   * RAMDocs
2. which of these are pilot-only vs main-benchmark datasets
3. whether initial contamination injection is manual, heuristic, or LLM-assisted
4. how much manual labeling budget to spend on pilot examples
5. whether instability is treated as a contamination subtype or a separate evaluation condition

---

# 23. Acceptance criteria

This PRD has succeeded if, after running PRD 1 systems on it, you can credibly say:

* “Here are clean cases where RAG works.”
* “Here are contaminated cases where RAG fails despite relevant support being present.”
* “Here are missing-evidence cases that fail for a different reason.”
* “These categories are consistent enough to drive controller design.”
* “These patterns appear not just in synthetic setups, but in natural contamination datasets including WikiContradict, FaithEval, AmbigDocs, RAGTruth, and RAMDocs.”

That is the real purpose of this stage.

