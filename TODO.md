# PRD 1 — Implementation TODO & Scratchpad

> Auto-maintained progress tracker for TDD implementation.
> Last updated: 2026-04-01

---

## Overall Progress

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

---

## Current Phase: COMPLETE ✅

All 20 phases delivered across 3 sessions.

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
