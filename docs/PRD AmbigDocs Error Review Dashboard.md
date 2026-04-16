# PRD: AmbigDocs Error Review Dashboard

## 1. Overview

Build a local, browser-based HTML dashboard for human review of AmbigDocs error cases. The dashboard will support manual validation and correction of an existing heuristic error decomposition for a RAG run.

The core analytical distinction the tool must preserve is:

* **Retrieval-limited failure**: the retrieved set does not contain enough support to fully answer the question.
* **Post-retrieval failure**: the retrieved set appears to contain enough support, but the model still fails to aggregate, disambiguate, select, or format the answer correctly.

The dashboard is not a general-purpose annotation platform. It is a focused review tool for this AmbigDocs error-analysis workflow.

## 2. Product Goal

Enable a human reviewer to inspect a question, gold answers, model answer, evaluator output, and retrieved passages, then assign corrected labels for:

* output type
* retrieval support
* primary failure cause
* secondary tags
* free-text notes

The dashboard should make it easy to compare heuristic labels against human labels and export reviewed annotations for downstream analysis.

## 3. Users

Primary user:

* researcher performing manual audit of RAG errors

Secondary user:

* second annotator or adjudicator reviewing the same examples

## 4. Inputs and Source Files

The dashboard must support these files.

### Default review dataset

* `ambigdocs_stratified_error_samples.jsonl`

### Supporting label definitions

* `ambigdocs_paper_ready_taxonomy.csv`
* `ambigdocs_error_annotation_rubric.md`

### Optional full raw run files

* `inputs.jsonl`
* `retrievals.jsonl`
* `predictions.jsonl`
* `evaluations.jsonl`
* `prompts.jsonl`

### Optional summary input

* `ambigdocs_error_summary.csv`

The dashboard should work even if only the stratified JSONL plus taxonomy/rubric files are loaded.

## 5. Canonical Label Schema

The dashboard must use the existing taxonomy and rubric as the source of truth. Do not invent a new taxonomy.

### 5.1 Output label

Allowed values:

* complete
* partial
* ambiguous
* merged
* wrong

### 5.2 Retrieval-support label

Allowed values:

* full_support_present
* partial_support_present
* no_support_present
* unclear

### 5.3 Primary failure cause

Allowed values:

* retrieval_missing_support
* omitted_supported_answers
* single_answer_collapse
* wrong_selection_or_shape
* unsupported_generation
* entity_merge
* other

### 5.4 Secondary tags

Allowed values should include:

* multi_answer_prompt_not_followed
* wrong_granularity
* normalization_or_formatting_issue
* abstained_despite_evidence
* retrieval_contains_only_subset_of_entities
* distractor_heavy_retrieval
* evaluator_mismatch_or_near_miss
* other

Also allow:

* free-text custom tags

### 5.5 Reviewer note

Free-text field.

## 6. Functional Requirements

## 6.1 File loading

The dashboard must support:

* drag-and-drop file upload
* file picker upload
* loading bundled default files if present in the same folder

Supported formats:

* JSONL
* JSON
* CSV
* Markdown for rubric display

The tool must parse and validate loaded files and show useful error messages if formatting is invalid.

## 6.2 Example browser

The dashboard must display one example at a time.

For each example, show:

* example ID
* question
* gold answers
* model prediction
* evaluator label, if present
* heuristic bucket
* prompt text, if present
* retrieval support diagnostics, if present
* short explanatory note, if present
* retrieved passages, grouped by rank
* document metadata when available:

  * rank
  * title
  * document/source ID
  * retriever/reranker metadata if available

The UI should separate these into visually clear panels.

## 6.3 Annotation panel

For the current example, the reviewer must be able to assign:

* output label
* retrieval-support label
* primary failure cause
* secondary tags
* reviewer notes

The panel must also show:

* existing heuristic/system labels
* current human labels
* whether the example has been reviewed
* whether human labels disagree with heuristic labels

## 6.4 Navigation

The dashboard must support:

* next/previous example
* jump to example ID
* jump to next unreviewed example
* jump to next disagreement example
* jump to next example in current filter set

## 6.5 Filtering and search

The dashboard must support filters for:

* heuristic bucket
* evaluator label
* human-reviewed vs unreviewed
* human primary cause
* human output label
* disagreement vs agreement
* presence of a particular secondary tag

The dashboard must support search by:

* question text
* entity name
* gold answer string
* predicted answer string
* document title if available

## 6.6 Keyboard shortcuts

Implement keyboard shortcuts for fast review. At minimum:

* next example
* previous example
* mark reviewed
* assign common output labels
* assign common support labels
* focus notes field
* save/export

Provide a visible help modal listing all shortcuts.

## 6.7 Persistence

The dashboard must autosave annotations in browser local storage.

The user must also be able to:

* manually save
* export annotations to JSON
* export annotations to CSV
* re-import a saved annotation file
* merge imported annotations with the current session

Local persistence should be robust across refreshes.

## 6.8 Summary and analytics

Include a summary view with:

* total examples
* reviewed count
* unreviewed count
* review completion percentage
* counts by heuristic bucket
* counts by human primary cause
* counts by human output label
* counts by retrieval-support label
* counts by secondary tags

Also include:

* confusion matrix: heuristic bucket vs human primary cause
* confusion matrix: evaluator label vs human output label
* high-level split:

  * retrieval-limited failures
  * post-retrieval failures
  * unclear/other

Simple tables are sufficient. Charts are nice-to-have.

## 6.9 Rubric and taxonomy display

The dashboard must expose:

* a readable rubric panel loaded from `ambigdocs_error_annotation_rubric.md`
* a taxonomy reference panel loaded from `ambigdocs_paper_ready_taxonomy.csv`

The reviewer should be able to consult label definitions without leaving the page.

## 7. UX Requirements

The dashboard should feel lightweight and fast.

### Layout

Suggested layout:

* left sidebar: filters, navigation, progress
* center main area: example content
* right sidebar: annotation controls and rubric shortcuts

### Passage presentation

Retrieved passages should:

* be shown in rank order
* have collapsible bodies
* support expand/collapse all
* preserve line breaks
* be easy to skim

### Disagreement visibility

If human labels differ from heuristic labels, visually flag the example.

### Review status

Each example should clearly show:

* unreviewed
* reviewed
* flagged for adjudication

## 8. Nice-to-Have Features

These are optional unless easy to add:

* highlight spans in passages matching gold answers
* highlight spans matching predicted answer
* compact evidence-support view summarizing which gold answers appear in retrieval
* adjudication mode for second annotator
* side-by-side comparison of two annotators
* ability to hide/show prompt text
* dark mode
* export filtered subset only

## 9. Non-Goals

Do not build:

* a backend service
* a database
* multi-user live sync
* a generic annotation platform
* automatic relabeling logic beyond displaying precomputed heuristic labels

Do not change:

* taxonomy definitions
* rubric semantics
* the retrieval-vs-post-retrieval conceptual split

## 10. Technical Constraints

* Must run locally in a browser
* No server required
* Prefer a single self-contained HTML file, or a minimal static bundle
* Must work by opening the file directly or serving as static files
* No external API dependencies
* No network calls required for core usage

Preferred stack:

* plain HTML/CSS/JS or lightweight framework
* avoid heavy dependencies unless clearly justified

## 11. Data Model

Each review record should minimally store:

* example_id
* source_dataset
* question
* gold_answers
* prediction
* evaluator_label
* heuristic_bucket
* prompt_text
* retrievals
* human_output_label
* human_retrieval_support
* human_primary_failure_cause
* human_secondary_tags
* reviewer_notes
* reviewed_boolean
* flagged_for_adjudication_boolean
* timestamp_last_modified

Keep the schema stable and exportable.

## 12. Acceptance Criteria

The dashboard is acceptable when all of the following are true:

1. A reviewer can load the stratified JSONL and taxonomy/rubric files locally.
2. A reviewer can inspect each example’s question, gold answers, prediction, and retrieved passages.
3. A reviewer can assign and edit all required human labels.
4. The tool autosaves state locally.
5. The tool exports reviewed annotations to JSON and CSV.
6. The tool supports filtering by heuristic bucket and reviewed status.
7. The tool shows summary counts and a heuristic-vs-human confusion view.
8. The rubric and taxonomy can be viewed inside the dashboard.
9. The tool runs without a server.
10. The core distinction between retrieval-limited and post-retrieval failures is preserved in the UI and summary.

## 13. Suggested Implementation Plan

### Phase 1: Core viewer

* load stratified JSONL
* display one example at a time
* show question, gold answers, prediction, heuristic labels, passages

### Phase 2: Annotation workflow

* add annotation controls
* add reviewed status
* add autosave with local storage

### Phase 3: Export/import

* export JSON and CSV
* import saved annotations
* merge saved annotations

### Phase 4: Filters and summary

* add search and filters
* add counts and confusion tables
* add progress indicators

### Phase 5: Quality-of-life improvements

* keyboard shortcuts
* collapsible passages
* rubric/taxonomy reference panel
* disagreement highlighting

## 14. Deliverables

The coding agent should produce:

* `index.html` or equivalent entry file
* any minimal supporting JS/CSS assets
* clear instructions for local use
* sample screenshot or short usage note
* annotation export format documentation

## 15. Handoff Note for the Coding Agent

Build a local HTML review dashboard for AmbigDocs error analysis. Use the provided taxonomy CSV and rubric Markdown as canonical label definitions, and the stratified JSONL as the default dataset. The reviewer must be able to inspect each example’s question, gold answers, prediction, evaluator output, and retrieved passages, then assign corrected labels for output type, retrieval support, primary failure cause, secondary tags, and notes. Preserve the key analytical split between retrieval-limited and post-retrieval failures. Include filtering, keyboard shortcuts, autosave, and export to JSON/CSV.
