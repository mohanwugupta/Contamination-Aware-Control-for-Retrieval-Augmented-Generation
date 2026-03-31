Absolutely — here is the **revised, fully folded PRD 4** with the **success rubric and submission thresholds integrated into the original document**. It preserves the original structure while adding concrete red / yellow / green criteria for deciding whether the results are strong enough for a serious submission push. 

---

# PRD 4 — Evaluation, Ablations, Paper Artifacts, and Submission Thresholds

## Project name

**Evaluation Framework for Contamination-Aware Control in RAG**

## Purpose

Design and execute the experimental campaign that evaluates whether the contamination-aware controller improves RAG reliability under contaminated retrieval conditions.

This stage is responsible for:

* formal comparisons,
* metric definition,
* ablation design,
* statistical reporting,
* case-study generation,
* final research artifacts for a paper or public release,
* and explicit decision thresholds for whether the project is strong enough for a serious submission.

## Why this stage exists

PRD 1 created the baseline RAG systems.
PRD 2 created the contamination benchmark suite.
PRD 3 created the controller.

PRD 4 determines whether the central claim actually holds:

> Modeling retrieved-set contamination explicitly improves RAG behavior under misleading evidence mixtures, especially by reducing confident hallucinations while preserving utility on clean retrieval.

This stage turns the project into something publishable.

---

# 1. Goals

## Primary goal

Evaluate whether the contamination-aware controller outperforms strong baseline RAG systems on contamination-focused benchmarks.

## Secondary goals

* identify which controller components matter most,
* measure tradeoffs between safety and utility,
* test robustness across contamination types,
* produce clear tables, plots, and case studies for writeup,
* generate a reproducible evaluation artifact,
* and determine whether the project is in red, yellow, or green territory for submission quality.

## Non-goals

This stage will **not**:

* redesign the controller,
* add new benchmark families unless absolutely necessary,
* expand scope into unrelated RAG problems,
* optimize endlessly for benchmark score without preserving the core scientific question.

---

# 2. Core evaluation questions

The experimental campaign should answer these questions explicitly.

## RQ1

Do standard RAG systems fail more often under **contaminated retrieval** than under clean retrieval, even when relevant evidence is present?

## RQ2

Does the contamination-aware controller reduce **hallucination rate** and especially **confident hallucination rate** relative to standard full-context RAG?

## RQ3

Does **minimal-consistent-subset answering** outperform:

* answering from all retrieved passages,
* naive reduced-context baselines,
* abstention-only control?

## RQ4

Does the controller preserve performance on **clean retrieval** rather than merely improving safety by over-abstaining?

## RQ5

Which contamination signals and routing mechanisms contribute most to any observed gains?

## RQ6

Are the effects consistent across:

* contradiction,
* ambiguity,
* partial-match lure,
* unsupported synthesis,
* instability,
* and missing-evidence conditions?

## RQ7

Are the gains strong enough, clean enough, and general enough to justify a serious NeurIPS-style submission push?

---

# 3. Success criteria

PRD 4 is complete if all of the following are true:

1. all planned baselines and controller variants are run on frozen dev/eval splits,
2. primary metrics are computed automatically,
3. contamination-type breakdowns are reported,
4. ablations isolate the value of major controller components,
5. case studies illustrate both successes and failures,
6. plots and tables are ready for direct use in a paper draft.

Preferred:
7. results are stable across at least two baseline RAG variants or two generator settings.

## Additional success rule

This project should not be judged by average QA gains alone. A strong result must show:

1. meaningful gains on the **target failure mode**,
2. gains that are **not explained mainly by abstention**,
3. little or no collapse on **clean retrieval**,
4. improvement over **strong simple baselines**, not just the weakest baseline.

---

# 4. Systems to evaluate

## Required systems

### System A — Standard full-context RAG

The baseline from PRD 1:

* retrieved top-k passages
* optional reranking
* answer from full retrieved set

### System B — Reduced-context baseline

A naive smaller-context baseline:

* answer from top-1 or top-2 only

Purpose:

* test whether simply using less context already helps

### System C — Abstention-only controller

Same contamination scoring as PRD 3, but no subset selection:

* low contamination → answer from full set
* high contamination → abstain

Purpose:

* isolate whether gains come just from abstention

### System D — Subset-only controller

Always use subset selection when available, but do not abstain unless impossible

Purpose:

* isolate the value of subset selection separately from abstention

### System E — Full contamination-aware controller

The full method:

* contamination scoring
* subset selection
* routing among full-set, subset, and abstain

## Optional stronger baselines

If resources allow:

* simple filtering baseline that removes the single most suspicious passage
* NLI-based filtering baseline
* stability-aware baseline using multi-order answer consistency
* corrective retrieval / retrieval-quality gating baseline

These are optional, but good to have if feasible.

---

# 5. Evaluation datasets and slices

PRD 4 should use the finalized outputs of PRD 2.

## Required evaluation slices

### Slice 1 — Clean retrieval

Purpose:

* measure normal utility
* ensure controller does not overfire

### Slice 2 — Contaminated retrieval

Purpose:

* test core hypothesis

### Slice 3 — Missing-evidence retrieval

Purpose:

* distinguish contamination handling from insufficiency handling

## Required subtype breakdowns within contaminated retrieval

* T1 conflicting answer-bearing detail
* T2 same-name / entity ambiguity
* T3 partial-match lure
* T4 unsupported synthesis
* T5 retrieval instability

## Required natural dataset reporting

Report results separately where possible on:

* WikiContradict
* FaithEval
* AmbigDocs
* RAGTruth
* RAMDocs

This matters because the paper should show the failure mode is not only synthetic.

---

# 6. Primary metrics

These are the metrics that should carry the results section.

## 6.1 Accuracy / task correctness

Standard answer correctness on answerable examples.

Use:

* exact match
* normalized match
* or task-appropriate factual scoring

## 6.2 Hallucination rate

Fraction of outputs that are factually incorrect relative to the gold answer or retrieved context standard.

## 6.3 Confident hallucination rate

Fraction of wrong answers produced with high confidence.

This should be a headline metric.

### Confidence definition

Must be fixed in advance. Options include:

* model self-reported confidence
* judge-based confidence label
* answer probability proxy if available
* heuristic high-confidence formatting or non-hedged response

Pick one and keep it consistent.

## 6.4 Abstention rate

Fraction of examples where the system declines to answer.

## 6.5 Selective accuracy

Accuracy conditional on answering.

This is crucial because it prevents wins from looking good just by refusing more often.

## 6.6 Coverage

Fraction of examples for which the system gives an answer.

Coverage should always be reported alongside selective accuracy.

## 6.7 Clean-set utility

Accuracy on clean retrieval slice.

The controller should not destroy this.

---

# 7. Secondary metrics

These support diagnosis and ablations.

## 7.1 Contamination detection accuracy

How well contamination score separates clean from contaminated examples.

Use:

* AUROC
* AUPRC
* F1 at threshold

## 7.2 Subset quality

How often the selected subset truly supports the final answer.

Possible components:

* subset correctness
* subset faithfulness
* subset precision

## 7.3 Average subset size

How many passages are retained in subset-answer mode.

## 7.4 Stability gain

Reduction in answer variance under perturbation for systems using the controller.

## 7.5 Auditability measures

Mostly qualitative, but may include:

* fraction of decisions with interpretable reason code
* fraction of subset decisions with non-empty rationale summary

## 7.6 Error-type distribution

Break down failures into:

* wrong answer with conflicting support present
* wrong answer under ambiguity
* wrong synthesis without coherent subset
* false abstention
* missed abstention

---

# 8. Core experiment suite

## Experiment 1 — Baseline vulnerability to contamination

### Question

Do baseline RAG systems fail more under contaminated retrieval than under clean retrieval?

### Systems

* Standard full-context RAG
* reduced-context baseline

### Conditions

* clean
* contaminated
* missing-evidence

### Expected result

* high accuracy on clean
* clear degradation on contaminated
* different failure profile on missing-evidence

### Purpose

This establishes the problem.

---

## Experiment 2 — Main controller comparison

### Question

Does the full controller reduce hallucinations relative to baseline RAG?

### Systems

* Standard full-context RAG
* abstention-only controller
* subset-only controller
* full controller

### Conditions

* especially contaminated retrieval
* also clean and missing-evidence

### Primary metrics

* hallucination rate
* confident hallucination rate
* selective accuracy
* coverage

### Expected result

The full controller should reduce confident hallucinations on contaminated examples without a major collapse on clean examples.

---

## Experiment 3 — Value of subset selection

### Question

Is minimal-consistent-subset answering better than naive context reduction?

### Systems

* top-1
* top-2
* simple passage-drop heuristic
* subset-only controller
* full controller

### Conditions

* contaminated examples with at least one coherent support subset

### Expected result

The controller’s subset selection should outperform naive “use less context” baselines.

---

## Experiment 4 — Contamination vs missing evidence

### Question

Does the controller treat contamination differently from insufficiency?

### Systems

* full-context baseline
* abstention-only controller
* full controller

### Conditions

* contaminated vs missing-evidence

### Expected result

The controller should:

* subset-answer more often in contamination cases with salvageable support
* abstain more often in truly missing-evidence cases

This is important for the paper’s conceptual distinction.

---

## Experiment 5 — Robustness by contamination subtype

### Question

Which contamination types are hardest, and where does the controller help most?

### Report by subtype

* T1 conflicting detail
* T2 ambiguity
* T3 partial-match lure
* T4 unsupported synthesis
* T5 instability

### Expected result

The controller may help more on contradiction and ambiguity than on unsupported synthesis early on. That is acceptable, as long as you report it honestly.

---

## Experiment 6 — Natural dataset generalization

### Question

Do effects hold on naturally occurring contamination datasets?

### Required reporting

Separate or grouped reporting on:

* WikiContradict
* FaithEval
* AmbigDocs
* RAGTruth
* RAMDocs

### Expected result

At least some measurable gain should appear beyond the synthetic/controlled slice.

---

# 9. Ablation plan

This section is essential.

## 9.1 Signal ablations

Turn off one signal group at a time:

* no candidate competition
* no cross-passage inconsistency
* no entity ambiguity
* no unsupported synthesis signal
* no perturbation instability

Purpose:

* determine which signals matter most

## 9.2 Routing ablations

Compare:

* no abstention
* no subset selection
* always subset when available
* full threshold router

Purpose:

* isolate the policy contribution

## 9.3 Threshold ablations

Vary:

* low threshold
* high threshold
* stricter vs looser abstention settings

Purpose:

* measure utility-safety tradeoff

## 9.4 Subset search ablations

Compare:

* greedy subset search
* candidate-group heuristic
* naive top-k reduction

Purpose:

* test whether the subset algorithm matters

## 9.5 Model-stack ablations

If feasible:

* with reranker vs without reranker
* generator A vs generator B

Purpose:

* show method is not brittle to one stack

---

# 10. Statistical analysis requirements

The project should not stop at raw percentages.

## Required reporting

* mean metrics over full eval split
* per-slice breakdowns
* paired comparisons where possible

## Recommended statistical procedures

* bootstrap confidence intervals
* paired permutation or McNemar-style tests for accuracy changes
* risk-coverage analysis for selective answering behavior

## Reporting rule

Every headline gain should be accompanied by:

* effect size
* uncertainty interval
* denominator / sample size

---

# 11. Plot requirements

These plots should be produced automatically.

## Required plots

### Plot 1 — Main metric bar chart

Compare systems on:

* hallucination rate
* confident hallucination rate
* selective accuracy
* coverage

### Plot 2 — Risk-coverage curve

Show tradeoff between:

* answering more often
* making mistakes

### Plot 3 — Condition breakdown

Performance on:

* clean
* contaminated
* missing-evidence

### Plot 4 — Contamination subtype breakdown

Performance by T1–T5 subtype

### Plot 5 — Ablation summary

Effect of removing each signal group

## Preferred plots

### Plot 6 — Calibration / confidence plot

If confident hallucination metric supports it

### Plot 7 — Subset size distribution

How often the controller uses 1, 2, 3 passages, etc.

### Plot 8 — Natural dataset results

Performance by:

* WikiContradict
* FaithEval
* AmbigDocs
* RAGTruth
* RAMDocs

---

# 12. Table requirements

## Required tables

### Table 1 — Main results

Rows:

* baseline
* reduced-context baseline
* abstention-only
* subset-only
* full controller

Columns:

* clean accuracy
* contaminated accuracy
* hallucination rate
* confident hallucination rate
* abstention rate
* selective accuracy
* coverage

### Table 2 — Contamination subtype results

Rows:

* systems

Columns:

* T1
* T2
* T3
* T4
* T5

### Table 3 — Ablation results

Rows:

* full controller
* minus one signal variants
* routing variants

Columns:

* main metrics

### Table 4 — Natural dataset results

Rows:

* systems

Columns:

* WikiContradict
* FaithEval
* AmbigDocs
* RAGTruth
* RAMDocs

---

# 13. Case study requirements

This project needs readable examples, not just aggregates.

## Required case-study categories

### Case type A — Baseline fails, controller succeeds via subset selection

Show:

* conflicting or lure-like retrieval
* chosen subset
* correct answer

### Case type B — Baseline fails, controller succeeds via abstention

Show:

* missing-evidence or unsalvageable contamination
* abstention reason

### Case type C — Controller failure

Show:

* false abstention
* wrong subset
* missed contamination
* subtype where method is still weak

### Case type D — Natural dataset example

Show at least one case from a natural contamination benchmark

## Required contents per case

* question
* retrieved passages or summaries
* contamination score breakdown
* baseline output
* controller output
* explanation

These should be ready to drop into a paper appendix or blog post.

---

# 14. Artifact requirements

## Code artifacts

* reproducible evaluation scripts
* ablation runner
* metrics computation scripts
* plotting scripts
* case-study export script

## Data artifacts

* frozen eval splits
* run outputs
* summary tables
* example packs

## Paper artifacts

* main tables
* plots
* case studies
* methods appendix material
* limitations summary

## Reproducibility requirements

Every result table should be reproducible from:

* config file
* run seed
* frozen benchmark split
* saved predictions

---

# 15. Implementation tasks

## Task group A: evaluation runner

* implement batch comparison runner across systems
* ensure outputs are aligned by example ID

## Task group B: metric computation

* compute primary metrics
* compute slice and subtype metrics
* compute selective accuracy and coverage

## Task group C: statistical reporting

* implement bootstrap CIs
* implement paired significance tests
* summarize uncertainty

## Task group D: plotting

* generate standard plots
* ensure plots are publication-ready

## Task group E: ablation framework

* run signal and routing ablations
* save results under clear naming scheme

## Task group F: case study exporter

* identify representative examples
* export human-readable summaries
* support appendix-ready formatting

## Task group G: final packaging

* compile result directory
* create result manifest
* create short experimental summary memo

---

# 16. Run order

Recommended execution order:

### Phase 1

Run baseline systems on frozen eval suite

### Phase 2

Run full controller on dev, then eval

### Phase 3

Run key comparisons:

* baseline
* abstention-only
* subset-only
* full controller

### Phase 4

Run ablations

### Phase 5

Produce plots, tables, and case studies

Do not run every optional ablation before confirming the main result exists.

---

# 17. Submission-readiness rubric and quantitative thresholds

## Purpose

Define concrete thresholds for judging whether results are:

* too weak for a serious submission push,
* promising but not yet strong enough,
* or strong enough to justify full writeup and submission effort.

## 17.1 Core philosophy

A strong result should show:

1. meaningful gains on the **target failure mode**,
2. gains that are **not explained mainly by abstention**,
3. little or no collapse on **clean retrieval**,
4. improvement over **simple baselines**, not just full-context RAG.

## 17.2 Headline target: contaminated retrieval performance

### Metric

Selective accuracy on contaminated retrieval, or equivalent contamination-focused answer-quality metric.

### Red

* less than **3 absolute points** improvement over standard full-context baseline
* or less than **2 points** over the strongest simple baseline

### Yellow

* **3 to 5 absolute points** improvement over full-context baseline
* and at least **2 to 3 points** over the strongest simple baseline

### Green

* **5+ absolute points** improvement over full-context baseline
* and at least **3+ points** over the strongest simple baseline

## 17.3 Confident hallucination reduction

### Metric

Relative reduction in confident hallucination rate on contaminated examples.

### Red

* less than **10% relative reduction**

### Yellow

* about **10–20% relative reduction**

### Green

* **20–30%+ relative reduction**

### Very strong

* **30%+ relative reduction** with modest coverage loss

## 17.4 Clean retrieval preservation

### Metric

Accuracy change on clean retrieval relative to full-context baseline.

### Red

* more than **2 absolute points worse**

### Yellow

* between **1 and 2 points worse**

### Green

* within **1 point** of baseline
* or better

## 17.5 Coverage / abstention discipline

### Red

* coverage drops sharply, especially without matched-coverage gains
* or gains disappear when comparing at matched coverage

### Yellow

* moderate coverage drop, but selective accuracy improves clearly

### Green

* small to moderate coverage drop
* and gains remain clear at matched or near-matched coverage

## 17.6 Beating simple baselines

### Must beat

* standard full-context RAG
* top-1 or top-2 reduced-context baseline
* abstention-only controller
* simple filtering baseline if included

### Red

* controller does not clearly beat top-1/top-2 or abstention-only

### Yellow

* beats most simple baselines, but only narrowly

### Green

* beats all major simple baselines by a visible margin
* especially on contamination-heavy slices

## 17.7 Natural dataset threshold expectations

### Red

* no visible gain on any natural dataset

### Yellow

* gains on one natural dataset, mixed elsewhere

### Green

* visible gains on at least two of:

  * WikiContradict
  * FaithEval
  * AmbigDocs
  * RAGTruth
  * RAMDocs

## 17.8 Minimum acceptable for serious paper draft

* at least **3 points absolute** improvement on contaminated-slice selective accuracy
* at least **10% relative reduction** in confident hallucinations
* no more than **2 points** clean-slice loss
* beats full-context baseline and at least one simple baseline

## 17.9 Strong target

* at least **5 points absolute** improvement on contaminated-slice selective accuracy
* at least **20% relative reduction** in confident hallucinations
* no more than **1 point** clean-slice loss
* beats abstention-only and naive top-k baselines

## 17.10 Excellent target

* at least **5–8 points absolute** improvement on contaminated-slice selective accuracy
* at least **25–30%+ relative reduction** in confident hallucinations
* negligible clean-slice loss
* strong natural-dataset gains
* consistent wins across major contamination subtypes

## 17.11 Project decision rules

### Continue aggressively if:

* contaminated-slice gain is already **5+ points** on dev
* confident hallucination reduction is **20%+ relative**
* clean-slice drop is minor
* simple baselines are clearly beaten

### Continue cautiously if:

* gains are around **3–5 points**
* some wins are real but the story is not yet clean
* natural dataset results are mixed

### Reconsider or pivot if:

* gains stay under **3 points**
* abstention-only is nearly as good
* top-1/top-2 does as well as the full controller
* clean-slice utility degrades too much

---

# 18. Exit criteria

PRD 4 is done when:

1. all required systems have been run on frozen benchmark splits,
2. primary metrics are computed and saved,
3. main comparison tables exist,
4. subtype and natural-dataset breakdowns exist,
5. at least one ablation table exists,
6. at least 8 to 12 strong case studies are exported,
7. figures are ready for a draft paper.

Preferred:
8. the results support a clear story about contamination as a distinct failure mode and subset selection as a useful intervention,
9. the results meet at least the **yellow** submission threshold,
10. the project has a clear judgment on whether it is red, yellow, or green for full submission.

---

# 19. Risks and mitigations

## Risk 1

The controller only improves by abstaining excessively.

### Mitigation

Report selective accuracy and coverage prominently; compare against abstention-only control.

## Risk 2

The subset selector adds little beyond top-1 or top-2 reduction.

### Mitigation

Include strong reduced-context baselines and compare directly.

## Risk 3

Results only hold on synthetic contamination.

### Mitigation

Report separate performance on WikiContradict, FaithEval, AmbigDocs, RAGTruth, and RAMDocs.

## Risk 4

Ablations are inconclusive because signals are too correlated.

### Mitigation

Prioritize a small number of interpretable signals and report qualitative differences.

## Risk 5

Too many experiments dilute the story.

### Mitigation

Anchor the paper on:

* one main result table,
* one main risk-coverage plot,
* one subtype table,
* a few strong case studies.

---

# 20. Recommended MVP evaluation package

The smallest strong version of PRD 4 is:

* baseline vs abstention-only vs subset-only vs full controller
* clean / contaminated / missing-evidence slices
* hallucination rate, confident hallucination rate, selective accuracy, coverage
* one subtype breakdown
* one natural dataset breakdown
* 3 to 5 ablations
* 8 to 10 case studies
* preliminary red / yellow / green classification

That is enough for a serious artifact.

---

# 21. Acceptance criteria

This PRD has succeeded if you can credibly say:

* “Standard RAG fails in a distinct way under contaminated retrieval.”
* “Our controller reduces confident hallucinations under that condition.”
* “The gain is not just because we answer less.”
* “Subset selection contributes beyond naive context reduction.”
* “The pattern appears on both controlled and natural contamination data.”
* “The results clear at least the yellow threshold for submission seriousness.”

That is the paper-level standard.

