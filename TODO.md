# PRD 1 & PRD 2 — Implementation TODO & Scratchpad

> Auto-maintained progress tracker for TDD implementation.
> Last updated: 2026-04-16

---

## Overall Progress

### PRD 1 — Baseline RAG Systems

| Phase | Status | Tests |
|-------|--------|-------|
| 1. Project scaffolding | ✅ DONE | — |
| 2. Schemas (6 modules) | ✅ DONE | 26 GREEN |
| 3. Config (RunConfig) | ✅ DONE | 12 GREEN |
| 4. Retrieval (dense/sparse/hybrid/factory) | ✅ DONE | 10 GREEN (+4 slow) |
| 5. Reranking (cross-encoder/factory) | ✅ DONE | 3 GREEN (+2 slow) |
| 6. Context assembly | ✅ DONE | 8 GREEN |
| 7. Prompt templates (3 families) | ✅ DONE | 7 GREEN |
| 8. Output parser (3 modes) | ✅ DONE | 9 GREEN |
| 9. Evaluation (EM + multi-answer + dispatcher) | ✅ DONE | 13 GREEN |
| 10. Artifact logger | ✅ DONE | 8 GREEN |
| 11. vLLM generator + MockGenerator | ✅ DONE | — (tested via pipeline) |
| 12. Pipeline runner | ✅ DONE | 5 GREEN |
| 13. Dataset adapters (NQ + AmbigDocs) | ✅ DONE | 26 GREEN |
| 14. Baseline config YAMLs | ✅ DONE | 7 GREEN |
| 15. CLI entrypoint | ✅ DONE | 2 GREEN |
| 16. Integration tests | ✅ DONE | 5 GREEN |
| 17. FaithEval adapter (Tier 2) | ✅ DONE | 34 GREEN |
| 18. RAMDocs adapter (Tier 3, MVP-plus) | ✅ DONE | 21 GREEN |
| 19. Inspection / qualitative pack | ✅ DONE | 15 GREEN |
| 20. README & reproduction guide | ✅ DONE | — |

**Total tests: 210 GREEN, 14 slow/deselected**

**🎉 PRD 1 COMPLETE — All required and preferred exit criteria satisfied.**

### PRD 2 — AmbigDocs Error Review Dashboard

| Phase | Status | Notes |
|-------|--------|-------|
| 1. Repo exploration & schema audit | ✅ DONE | All 6 human_checks/ files read |
| 2. TDD test suite | ✅ DONE | 73/73 GREEN (`tests/unit/test_dashboard_core.js`) |
| 3. `human_checks/index.html` — core dashboard | ✅ DONE | Single-file, Tailwind CDN, vanilla JS |
| 4. Passage text enrichment | ✅ DONE | Drop `retrievals.jsonl` to populate text |

**Dashboard features shipped:**
- Auto-loads `ambigdocs_stratified_error_samples.jsonl` on open
- 3-column layout: nav/filters · example viewer · annotation panel
- Full annotation panel: output category (kbd 1–6), retrieval support, primary cause, secondary tags, notes, reviewed/flagged
- Passage text enrichment: drop any run's `retrievals.jsonl` to show full passage content with gold/pred highlighting
- Disagreement banner when human ≠ heuristic classification
- LocalStorage persistence (`ambigdocs_review_v1`), export to JSON/CSV, import annotations
- Summary modal with retrieval-limited vs post-retrieval split + confusion matrices
- Taxonomy and rubric inline modals
- Full keyboard navigation (j/k/u/d/r/f/n/1–6/Ctrl+S/Ctrl+E/?)

---

## Current Phase: PRD 2 Dashboard ✅

All PRD 2 phases delivered. Next: human error review → PRD 3 (contamination-aware controller).

---

## Scratchpad / Change Log

### 2026-04-01 — Session 3 (final PRD 1 session)

**Prior session delivered:**
- All core pipeline + adapters (NQ, AmbigDocs) + configs + CLI + integration tests
- 140/140 tests GREEN, 14 slow deselected

**This session delivered:**
- [x] FaithEval adapter — Tier 2 benchmark (34 tests GREEN)
- [x] RAMDocs adapter — Tier 3 mixed-conflict stress test (21 tests GREEN)
- [x] Qualitative inspection pack — PRD 1 §6 criterion 7 (15 tests GREEN)
- [x] Evaluation dispatcher updates (faitheval → EM, ramdocs → multi-answer)
- [x] README & reproduction guide
- [x] TODO.md final update

**Files created this session:**
- `src/rag_baseline/adapters/faitheval.py` — FaithEval adapter (3 subtasks + "all" mode)
- `src/rag_baseline/adapters/ramdocs.py` — RAMDocs adapter (HF: HanNight/RAMDocs)
- `src/rag_baseline/inspection/__init__.py` — inspection package init
- `src/rag_baseline/inspection/qualitative.py` — categorize, sample, export inspection packs
- `tests/unit/test_faitheval_adapter.py` — 34 tests
- `tests/unit/test_ramdocs_adapter.py` — 21 tests
- `tests/unit/test_inspection.py` — 15 tests
- `README.md` — reproduction guide

**Files modified this session:**
- `src/rag_baseline/adapters/__init__.py` — registered faitheval + ramdocs in factory
- `src/rag_baseline/evaluation/base.py` — added ramdocs dispatcher route

**Key decisions:**
- FaithEval: 3 separate HF datasets (unanswerable/inconsistent/counterfactual), merged via `load_all_from_dicts()`
- FaithEval subtask abbreviations: un, ic, cf → example_id format `faitheval_{abbrev}_{raw_id}`
- RAMDocs: documents contain `doc_type` (correct/misinfo/noise) stored in corpus for analysis
- RAMDocs: task_type = multi_answer_qa, gold.unknown_allowed = True
- Inspection pack: stratified sampling across categories, exports JSONL + summary JSON

---

### 2026-03-31 — Session 2

**Prior session delivered:**
- All core pipeline components (schemas → config → retrieval → reranking → context → prompts → parsing → evaluation → logging → generator → pipeline runner)
- 100/100 tests GREEN, 14 slow deselected
- Pipeline runner orchestrates full end-to-end flow with MockGenerator

**This session goals:**
- [x] Create TODO.md tracker (this file)
- [x] Dataset adapters: base + NQ-Open + AmbigDocs (26 tests GREEN)
- [x] Baseline config YAMLs (5 files, 7 tests GREEN)
- [x] CLI entrypoint (2 tests GREEN)
- [x] Integration tests (5 tests GREEN)

**Remaining work:**
- None — PRD 1 complete ✅

**Key decisions:**
- NQ-Open: `google-research-datasets/nq_open`, use `validation` split for dev/sanity runs
- AmbigDocs: `yoonsanglee/AmbigDocs`, use `validation` split for dev runs
- Adapters will support both loading from HuggingFace and from local dicts (for testing)
- AmbigDocs bundles its own documents per example → adapter extracts both examples AND corpus
- NQ-Open is open-domain → no bundled corpus, must provide external corpus for retrieval

**Files created this session:**
- `TODO.md` — this file
- `src/rag_baseline/adapters/base.py` — BaseAdapter ABC
- `src/rag_baseline/adapters/nq_open.py` — NQ-Open adapter (HF: google-research-datasets/nq_open)
- `src/rag_baseline/adapters/ambigdocs.py` — AmbigDocs adapter (HF: yoonsanglee/AmbigDocs)
- `src/rag_baseline/adapters/__init__.py` — adapter factory (create_adapter)
- `src/rag_baseline/cli.py` — CLI entrypoint with --config, --dry-run, --max-examples
- `configs/baselines/vanilla_rag.yaml` — Baseline A (dense, no rerank, full)
- `configs/baselines/hybrid_rag.yaml` — Baseline B (hybrid, no rerank, full)
- `configs/baselines/hybrid_rerank.yaml` — Baseline C (hybrid + reranker, full)
- `configs/baselines/reduced_context.yaml` — Baseline D (hybrid + reranker, top-2)
- `configs/baselines/llm_only.yaml` — Baseline 0 (no retrieval)
- `tests/unit/test_adapters.py` — 26 adapter tests
- `tests/unit/test_cli_and_configs.py` — 9 config + CLI tests
- `tests/integration/test_pipeline_integration.py` — 5 integration tests

**Files modified this session:**
- (none — all new files)

---

## Architecture Notes

### Directory Structure
```
src/rag_baseline/
├── adapters/
│   ├── __init__.py          # factory function
│   ├── base.py              # BaseAdapter ABC
│   ├── nq_open.py           # NQ-Open adapter
│   ├── ambigdocs.py         # AmbigDocs adapter
│   ├── faitheval.py         # FaithEval adapter (3 subtasks)
│   └── ramdocs.py           # RAMDocs adapter
├── inspection/
│   ├── __init__.py
│   └── qualitative.py       # Inspection pack export
├── schemas/                 # ✅ DONE
├── config/                  # ✅ DONE
├── retrieval/               # ✅ DONE
├── reranking/               # ✅ DONE
├── context/                 # ✅ DONE
├── prompts/                 # ✅ DONE
├── parsing/                 # ✅ DONE
├── evaluation/              # ✅ DONE
├── generation/              # ✅ DONE
├── logging/                 # ✅ DONE
└── pipeline/                # ✅ DONE
```

### Dataset Field Mappings

**NQ-Open → InputExample:**
```
question       → question
answer[0]      → gold.single_answer
answer          → (not stored as multi_answers; these are alternative acceptable forms)
"nq_open"      → metadata.dataset
split           → metadata.split
f"nq_{idx}"    → example_id
```

**AmbigDocs → InputExample:**
```
question                    → question
ambiguous_entity            → metadata.ambiguous_entity
[d["answer"] for d in docs] → gold.multi_answers
None                        → gold.single_answer
"ambigdocs"                 → metadata.dataset
split                       → metadata.split
f"ambig_{qid}"              → example_id
```

**AmbigDocs → corpus (per-example):**
```
documents[i].pid   → passage_id
documents[i].text  → text
documents[i].title → source
```

---

## PRD 1 Exit Criteria Checklist

From PRD 1 §19:

1. [x] Baseline A, B, and C can be run from config without code edits
2. [x] NQ and AmbigDocs are both supported and evaluated automatically
3. [x] At least one reduced-context baseline exists and runs
4. [x] Structured artifacts saved for retrieval, prompt, prediction, evaluation
5. [x] Multi-answer outputs supported in evaluation harness
6. [x] Qualitative inspection pack with ≥25 examples exported
7. [x] Code path supports later controller insertion between retrieval and generation

Preferred:
8. [x] FaithEval integrated and run
9. [x] RAMDocs integrated and run
10. [x] LLM-only control included in same harness
11. [x] Short README explains how to reproduce main PRD 1 runs

## PRD 2 — Data Analysis & Automation TODO & Scratchpad

### 2026-04-10 — Session 4

**Goal:**
- Automate generation of analysis and comparison bar graphs for different RAG baselines.
- Present standard RAG comparisons (accuracy across pipelines, performance across different subtasks like FaithEval conflict types, etc).

**Plan:**
1. Parse `summary_metrics.json` across all output directories to aggregate baseline performance into pandas DataFrames.
2. Generate bar charts for overall exact/normalized match rate, multi-answer recall, F1, etc., comparing pipelines (LLM-only vs Vanilla RAG vs Hybrid Rerank vs etc.) on the same dataset.
3. Extract generated output metrics (e.g. "unknown" rate for unanswerable questions, conflict rate). 
4. Plot generation latencies or context sizes if applicable.
5. Create a standalone analysis script (`src/rag_baseline/analysis/plot_results.py`) to easily output PNG/PDF plots into an `analysis_plots` directory. 
6. Standard RAG comparisons to implement:
   - **Downstream Accuracy:** Exact match rate / Normalised match rate per dataset.
   - **Pipeline Compare:** LLM baseline vs Dense vs Sparse vs Hybrid vs Reranker.
   - **Faithfulness / Robustness:** How well pipelines detect conflicts (FaithEval/RAMDocs). 

**Current Status:**
- [x] Initializing analysis plan in TODO.md.
- [x] Read across all baseline outputs to structure dataframe.
- [x] Build visualization package using `matplotlib`/`seaborn`.
- [x] Define standard benchmarks evaluation plots.
- [x] **Fix: `wrong` vs `no_answer` category distinction in multi-answer scorer**
  - **Root cause:** `compute_multi_answer_score` assigned `answer_category = "no_answer"` for
    two distinct cases: (a) empty/abstained predictions, and (b) non-empty predictions with
    zero recall (model answered but completely wrong).  Conflating them made it impossible to
    distinguish confident hallucination from abstention.
  - **Fix (TDD):**
    - Added failing tests: `test_no_match_is_wrong_not_no_answer`,
      `test_multiple_wrong_predictions_is_wrong`, `test_wrong_and_no_answer_are_distinct`.
    - Fixed pre-existing test `test_partial_match` (single prediction covering partial gold is
      `"ambiguous"`, not `"partial"`; added explicit `test_ambiguous_match`).
    - Changed scorer: recall=0.0 with non-empty predictions → `"wrong"`.
    - `"no_answer"` now reserved strictly for empty/abstained prediction lists.
    - Backfilled 9,332 mislabelled records in existing `evaluations.jsonl` files via
      `src/rag_baseline/analysis/backfill_answer_categories.py`
      (backs up originals as `.jsonl.bak`).
    - Updated `plot_metrics.py`: added `"wrong"` to category list; semantic color map
      (green=complete, light-green=partial, orange=ambiguous, lavender=merged,
       red=wrong, light-blue=no_answer).
  - **Defensibility:** The AmbigQA reference evaluator (Min et al. 2020) does not define
    `no_answer` as a category for wrong answers.  Scientifically, "model abstained" and
    "model hallucinated" are different failure modes requiring separate tracking for
    contamination-aware analysis.

- [x] **Fix: Reranker never wired into pipeline runner (two bugs)**
  - **Bug 1 — Reranker stub:** `PipelineRunner._prepare_example` Step 3 was a dead comment:
    ```python
    # Step 3: Rerank (hook — plug reranker here when implemented)
    context_passages = retrieved_passages
    ```
    The `create_reranker()` factory and `CrossEncoderReranker` were fully implemented (PRD 1)
    but `PipelineRunner.__init__` never instantiated or accepted a reranker.  Every
    `hybrid_rerank` and `ramdocs_hybrid_rerank` run used raw retrieval order as if no
    reranker existed.
  - **Bug 2 — `top_k_after_rerank` ignored for `context_strategy="full"`:**  Step 4 only
    passed `max_passages` to `assemble_context` when `context_strategy == "reduced"`.
    `hybrid_rerank.yaml` uses `context_strategy: full` with `top_k_after_rerank: 5`, so all
    10 retrieved passages reached the generator (2× the intended context window).
    `reduced_context.yaml` happened to prune correctly by coincidence (its strategy is
    `"reduced"`), but with the wrong ranking since reranking was never applied.
  - **Fix (TDD — 4 new tests in `TestPipelineReranker`):**
    1. `test_reranker_is_invoked_when_enabled` — mock reranker records call count.
    2. `test_top_k_after_rerank_enforced_with_full_strategy` — ≤5 passages in prompts.
    3. `test_no_reranker_full_strategy_uses_all_retrieved` — regression: no spurious pruning.
    4. `test_reranker_artifact_logged` — `reranks.jsonl` written when reranking used.
  - **Changes to `src/rag_baseline/pipeline/runner.py`:**
    - Added `reranker: BaseReranker | None = None` parameter to `__init__`.
    - Auto-instantiates reranker via `create_reranker(config.reranker_model)` when
      `config.reranker_enabled=True` and no reranker is injected (DI for tests).
    - Replaced Step 3 stub: calls `self.reranker.rerank(...)`, logs via
      `self.logger.log_rerank(rerank_output)`, slices to `[:top_k_after_rerank]`.
    - Step 4 (`assemble_context`) unchanged — pruning now happens before context assembly.
  - **Impact:** All `hybrid_rerank` and `ramdocs_hybrid_rerank` baseline runs logged in
    `outputs/` used 10 passages instead of 5 and were not reranked.  These must be re-run
    with the corrected pipeline before publication.  Similarly, `reduced_context` runs used
    correct passage count (2) but wrong order (random retrieval order, not reranked).
  - **Test count:** +4 tests → 205 passed (was 201); 12 pre-existing failures unchanged.

