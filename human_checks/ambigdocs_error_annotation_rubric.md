# AmbigDocs Error Annotation Rubric (for this run)

This rubric is designed for the uploaded AmbigDocs validation run with hybrid retrieval + reranking + Qwen2.5-32B-Instruct. The goal is to separate **retrieval failures** from **post-retrieval reasoning failures**.

## Recommended annotation fields

For each example, record:

1. `output_category`
2. `retrieval_support`
3. `primary_failure_cause`
4. `secondary_tags`
5. `notes`

## 1) Output category

Use exactly one primary output label:

- **complete**  
  The model returns all valid answers required by the example.

- **single_answer_collapse**  
  The question requires multiple valid answers, but the model gives only one answer or one referent.

- **partial_recall**  
  The model returns more than one valid answer, but misses at least one supported gold answer.

- **entity_merge**  
  The model blends distinct entities, senses, or answer types into a mixed response.

- **wrong_selection_or_shape**  
  The output is grounded in retrieved content, but still fails because it picks the wrong entity, wrong granularity, wrong framing, or an evaluator-sensitive answer form.

- **unsupported_generation**  
  The model generates unsupported content, or abstains/claims insufficiency despite relevant evidence being present.

## 2) Retrieval support

Assess whether the retrieved set contains support for the gold answers.

- **all_gold_supported**  
  Every gold answer appears to be supported by at least one retrieved passage.

- **partial_gold_supported**  
  Only some gold answers appear in the retrieved set.

- **no_gold_supported**  
  None of the gold answers appear in the retrieved set.

This field should be annotated independently from model behavior.

## 3) Primary failure cause

Choose the best single explanation:

- **retrieval_missing_support**  
  The model could not fully succeed because one or more gold answers are absent from retrieval.

- **omitted_supported_answers**  
  All needed evidence is present, but the model lists only a subset of the valid answers.

- **wrong_entity_selection**  
  Evidence is present, but the model locks onto the wrong sense or referent.

- **answer_shape_or_normalization**  
  The model is close, but answer form, over-specific phrasing, extra unsupported modifiers, or formatting causes the miss.

- **entity_blending**  
  The model mixes properties or answers across distinct entities/senses.

- **unsupported_inference_or_abstention**  
  The model invents content not supported by the passages, or incorrectly says the answer cannot be determined.

## 4) Secondary tags

Add any that apply:

- `multi_answer_prompt_not_followed`
- `extra_distractor_answers`
- `wrong_granularity`
- `wrong_time_slice`
- `wrong_location_slice`
- `formatting_noise`
- `document_reference_noise`
- `abstained_despite_evidence`
- `retrieval_contains_distractor_entities`
- `retrieval_contains_only_subset_of_entities`

## 5) Decision procedure

Use this order:

### Step A: Check retrieval support
Ask: *Could a perfect generator have produced the full gold set from the retrieved passages?*

- If no -> `retrieval_missing_support`
- If yes -> continue

### Step B: Check answer count and coverage
Ask: *Did the model return all valid answers?*

- If yes -> `complete`
- If exactly one answer for a multi-answer case -> `single_answer_collapse`
- If some but not all -> `partial_recall`

### Step C: Distinguish partial vs wrong
If the answer is not complete:
- If answers are supported but incomplete -> `omitted_supported_answers`
- If answers are supported but misframed, over-specific, or evaluator-sensitive -> `answer_shape_or_normalization`
- If answers are unsupported or the model abstains incorrectly -> `unsupported_inference_or_abstention`
- If senses are blended -> `entity_blending`

## Practical interpretation for this run

The strongest high-level buckets for this run are:

- retrieval missing support
- support present but omitted valid answers
- support present but wrong selection/shape
- single-answer collapse
- unsupported generation
- entity merge

These buckets are meant to support later intervention design:
- retrieval fixes
- better multi-answer prompting/decoding
- post-retrieval aggregation
- answer normalization
- conflict/disambiguation-aware generation
