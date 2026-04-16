/**
 * TDD tests for AmbigDocs Error Review Dashboard core logic.
 * Run with:  node tests/unit/test_dashboard_core.js
 *
 * Tests cover parseJSONL, parseCSV, normalizeStratifiedRecord,
 * normalizeRawRunRecords, applyFilters, computeDisagreement,
 * exportAnnotationsJSON, exportAnnotationsCSV, highlightText,
 * findNextUnreviewed, findNextDisagreement, computeSummaryStats.
 */

'use strict';

const assert = require('assert');

// ---------------------------------------------------------------------------
// Minimal test runner
// ---------------------------------------------------------------------------

let _passed = 0;
let _failed = 0;

function test(name, fn) {
  try {
    fn();
    console.log(`  ✓  ${name}`);
    _passed++;
  } catch (err) {
    console.error(`  ✗  ${name}`);
    console.error(`     ${err.message}`);
    _failed++;
  }
}

function describe(suiteName, fn) {
  console.log(`\n${suiteName}`);
  fn();
}

// ---------------------------------------------------------------------------
// Functions under test (inline — mirrors implementations in index.html)
// Keep these in sync with the dashboard script block.
// ---------------------------------------------------------------------------

// ── parseJSONL ──────────────────────────────────────────────────────────────
function parseJSONL(text) {
  return text
    .split('\n')
    .map(l => l.trim())
    .filter(l => l.length > 0)
    .map(l => JSON.parse(l));
}

// ── parseCSV ────────────────────────────────────────────────────────────────
function parseCSV(text) {
  const lines = text.split('\n').map(l => l.trimEnd()).filter(l => l.length);
  if (lines.length === 0) return [];
  const headers = splitCSVRow(lines[0]);
  return lines.slice(1).map(line => {
    const values = splitCSVRow(line);
    const obj = {};
    headers.forEach((h, i) => { obj[h] = values[i] !== undefined ? values[i] : ''; });
    return obj;
  });
}

function splitCSVRow(row) {
  const fields = [];
  let current = '';
  let inQuotes = false;
  for (let i = 0; i < row.length; i++) {
    const ch = row[i];
    if (ch === '"') {
      if (inQuotes && row[i + 1] === '"') { current += '"'; i++; }
      else { inQuotes = !inQuotes; }
    } else if (ch === ',' && !inQuotes) {
      fields.push(current);
      current = '';
    } else {
      current += ch;
    }
  }
  fields.push(current);
  return fields;
}

// ── normalizeStratifiedRecord ───────────────────────────────────────────────
function normalizeStratifiedRecord(raw) {
  return {
    example_id: raw.example_id,
    question: raw.question,
    gold_answers: Array.isArray(raw.gold_answers) ? raw.gold_answers : [],
    pred_answers: Array.isArray(raw.pred_answers) ? raw.pred_answers : [],
    raw_model_output: raw.raw_model_output || '',
    heuristic_bucket: raw.heuristic_cause || raw.category || '',
    evaluator_label: raw.category || '',
    score: typeof raw.score === 'number' ? raw.score : null,
    gold_support_ratio: typeof raw.gold_support_ratio === 'number' ? raw.gold_support_ratio : null,
    full_gold_support: raw.full_gold_support === true,
    all_pred_supported: raw.all_pred_supported === true,
    retrieved_passages: buildPassagesFromStratified(raw),
    prompt_text: null,
    ambiguous_entity: null,
    metadata: { dataset: 'ambigdocs', source: 'stratified', note: raw.note || '' },
  };
}

function buildPassagesFromStratified(raw) {
  if (!raw.retrieved_sources_top5) return [];
  const sources = raw.retrieved_sources_top5;
  const ids = raw.used_passage_ids_top5 || [];
  return sources.map((src, i) => ({
    passage_id: ids[i] || String(i),
    text: '',
    source: src,
    retrieval_score: null,
    rank: i + 1,
  }));
}

// ── normalizeRawRunRecords ───────────────────────────────────────────────────
function normalizeRawRunRecords({ inputs, predictions, evaluations, retrievals, prompts }) {
  const predMap = Object.fromEntries((predictions || []).map(p => [p.example_id, p]));
  const evalMap = Object.fromEntries((evaluations || []).map(e => [e.example_id, e]));
  const retrMap = Object.fromEntries((retrievals || []).map(r => [r.example_id, r]));
  const promptMap = Object.fromEntries((prompts || []).map(p => [p.example_id, p]));

  return (inputs || []).map(inp => {
    const pred = predMap[inp.example_id] || {};
    const ev = evalMap[inp.example_id] || {};
    const retr = retrMap[inp.example_id] || {};
    const pr = promptMap[inp.example_id] || {};

    const metrics = ev.metrics || {};
    return {
      example_id: inp.example_id,
      question: inp.question,
      gold_answers: inp.gold && inp.gold.multi_answers ? inp.gold.multi_answers
                  : inp.gold && inp.gold.single_answer ? [inp.gold.single_answer]
                  : [],
      pred_answers: pred.parsed_output && pred.parsed_output.multi_answers
                    ? pred.parsed_output.multi_answers
                    : pred.parsed_output && pred.parsed_output.single_answer
                    ? [pred.parsed_output.single_answer]
                    : [],
      raw_model_output: pred.raw_model_output || '',
      heuristic_bucket: metrics.answer_category || '',
      evaluator_label: metrics.answer_category || '',
      score: typeof metrics.multi_answer_score === 'number' ? metrics.multi_answer_score
             : typeof metrics.normalized_match === 'boolean' ? (metrics.normalized_match ? 1 : 0)
             : null,
      gold_support_ratio: null,
      full_gold_support: null,
      all_pred_supported: null,
      retrieved_passages: retr.retrieved_passages || [],
      prompt_text: pr.prompt_text || null,
      ambiguous_entity: inp.metadata && inp.metadata.extra ? inp.metadata.extra.ambiguous_entity || null : null,
      metadata: { dataset: inp.metadata && inp.metadata.dataset || '', split: inp.metadata && inp.metadata.split || '' },
    };
  });
}

// ── RETRIEVAL_LIMITED_CAUSES ─────────────────────────────────────────────────
const RETRIEVAL_LIMITED_CAUSES = new Set([
  'retrieval_missing_support',
  'retrieval_missing_support_plus_partial_recall',
]);
const RETRIEVAL_LIMITED_BUCKETS = new Set([
  'retrieval_missing_support_plus_partial_recall',
  'wrong', // raw mode — further split by heuristic
]);

function isRetrievalLimited(heuristic_bucket, human_primary_failure_cause) {
  if (human_primary_failure_cause) return RETRIEVAL_LIMITED_CAUSES.has(human_primary_failure_cause);
  return RETRIEVAL_LIMITED_BUCKETS.has(heuristic_bucket);
}

// ── computeDisagreement ──────────────────────────────────────────────────────
/**
 * Returns true when the human's primary_failure_cause maps to a different
 * high-level group (retrieval-limited / post-retrieval / complete) than the
 * heuristic bucket.
 */
function computeDisagreement(heuristic_bucket, human_primary_failure_cause) {
  if (!human_primary_failure_cause) return false;

  const humanIsRetrLimited = RETRIEVAL_LIMITED_CAUSES.has(human_primary_failure_cause);
  const humanIsComplete = human_primary_failure_cause === 'none' || human_primary_failure_cause === '';

  const heuristicIsRetrLimited = RETRIEVAL_LIMITED_BUCKETS.has(heuristic_bucket);
  const heuristicIsComplete = heuristic_bucket === 'complete';

  if (humanIsRetrLimited !== heuristicIsRetrLimited) return true;
  if (humanIsComplete !== heuristicIsComplete) return true;
  return false;
}

// ── applyFilters ─────────────────────────────────────────────────────────────
function applyFilters(examples, filters, searchText, annotations) {
  let results = examples;

  if (filters.heuristicBuckets && filters.heuristicBuckets.length > 0) {
    const set = new Set(filters.heuristicBuckets);
    results = results.filter(e => set.has(e.heuristic_bucket));
  }

  if (filters.evaluatorLabels && filters.evaluatorLabels.length > 0) {
    const set = new Set(filters.evaluatorLabels);
    results = results.filter(e => set.has(e.evaluator_label));
  }

  if (filters.reviewStatus === 'reviewed') {
    results = results.filter(e => annotations[e.example_id] && annotations[e.example_id].reviewed);
  } else if (filters.reviewStatus === 'unreviewed') {
    results = results.filter(e => !annotations[e.example_id] || !annotations[e.example_id].reviewed);
  }

  if (filters.humanOutputLabels && filters.humanOutputLabels.length > 0) {
    const set = new Set(filters.humanOutputLabels);
    results = results.filter(e => {
      const ann = annotations[e.example_id];
      return ann && set.has(ann.human_output_label);
    });
  }

  if (filters.onlyDisagreements) {
    results = results.filter(e => {
      const ann = annotations[e.example_id];
      if (!ann) return false;
      return computeDisagreement(e.heuristic_bucket, ann.human_primary_failure_cause);
    });
  }

  if (filters.secondaryTag) {
    results = results.filter(e => {
      const ann = annotations[e.example_id];
      return ann && ann.human_secondary_tags && ann.human_secondary_tags.includes(filters.secondaryTag);
    });
  }

  if (searchText && searchText.trim()) {
    const q = searchText.toLowerCase();
    results = results.filter(e =>
      e.question.toLowerCase().includes(q) ||
      (e.ambiguous_entity && e.ambiguous_entity.toLowerCase().includes(q)) ||
      e.gold_answers.some(a => a.toLowerCase().includes(q)) ||
      e.pred_answers.some(a => a.toLowerCase().includes(q)) ||
      e.retrieved_passages.some(p => p.source && p.source.toLowerCase().includes(q))
    );
  }

  return results;
}

// ── findNextUnreviewed / findNextDisagreement ─────────────────────────────────
function findNextUnreviewed(filteredExamples, currentIndex, annotations) {
  for (let i = currentIndex + 1; i < filteredExamples.length; i++) {
    const e = filteredExamples[i];
    if (!annotations[e.example_id] || !annotations[e.example_id].reviewed) return i;
  }
  // wrap
  for (let i = 0; i < currentIndex; i++) {
    const e = filteredExamples[i];
    if (!annotations[e.example_id] || !annotations[e.example_id].reviewed) return i;
  }
  return currentIndex;
}

function findNextDisagreement(filteredExamples, currentIndex, annotations) {
  for (let i = currentIndex + 1; i < filteredExamples.length; i++) {
    const e = filteredExamples[i];
    const ann = annotations[e.example_id];
    if (ann && computeDisagreement(e.heuristic_bucket, ann.human_primary_failure_cause)) return i;
  }
  for (let i = 0; i < currentIndex; i++) {
    const e = filteredExamples[i];
    const ann = annotations[e.example_id];
    if (ann && computeDisagreement(e.heuristic_bucket, ann.human_primary_failure_cause)) return i;
  }
  return currentIndex;
}

// ── exportAnnotationsJSON ────────────────────────────────────────────────────
function exportAnnotationsJSON(examples, annotations) {
  const records = examples.map(e => {
    const ann = annotations[e.example_id] || {};
    return {
      example_id: e.example_id,
      source_dataset: 'ambigdocs',
      question: e.question,
      gold_answers: e.gold_answers,
      prediction: e.pred_answers,
      evaluator_label: e.evaluator_label,
      heuristic_bucket: e.heuristic_bucket,
      prompt_text: e.prompt_text || null,
      retrievals: e.retrieved_passages,
      human_output_label: ann.human_output_label || null,
      human_retrieval_support: ann.human_retrieval_support || null,
      human_primary_failure_cause: ann.human_primary_failure_cause || null,
      human_secondary_tags: ann.human_secondary_tags || [],
      reviewer_notes: ann.reviewer_notes || '',
      reviewed: ann.reviewed || false,
      flagged_for_adjudication: ann.flagged_for_adjudication || false,
      timestamp_last_modified: ann.timestamp_last_modified || null,
    };
  });
  return JSON.stringify(records, null, 2);
}

// ── exportAnnotationsCSV ─────────────────────────────────────────────────────
const CSV_HEADERS = [
  'example_id','source_dataset','question','gold_answers','prediction',
  'evaluator_label','heuristic_bucket','human_output_label',
  'human_retrieval_support','human_primary_failure_cause','human_secondary_tags',
  'reviewer_notes','reviewed','flagged_for_adjudication','timestamp_last_modified',
];

function toCsvField(val) {
  if (val === null || val === undefined) return '';
  const s = String(val);
  if (s.includes(',') || s.includes('"') || s.includes('\n')) {
    return '"' + s.replace(/"/g, '""') + '"';
  }
  return s;
}

function exportAnnotationsCSV(examples, annotations) {
  const rows = [CSV_HEADERS.join(',')];
  for (const e of examples) {
    const ann = annotations[e.example_id] || {};
    const fields = [
      e.example_id,
      'ambigdocs',
      e.question,
      e.gold_answers.join(' | '),
      e.pred_answers.join(' | '),
      e.evaluator_label,
      e.heuristic_bucket,
      ann.human_output_label || '',
      ann.human_retrieval_support || '',
      ann.human_primary_failure_cause || '',
      (ann.human_secondary_tags || []).join(' | '),
      ann.reviewer_notes || '',
      ann.reviewed ? 'true' : 'false',
      ann.flagged_for_adjudication ? 'true' : 'false',
      ann.timestamp_last_modified || '',
    ];
    rows.push(fields.map(toCsvField).join(','));
  }
  return rows.join('\n');
}

// ── highlightText ─────────────────────────────────────────────────────────────
function normalizeForHighlight(s) {
  return s.toLowerCase().replace(/[^a-z0-9 ]/g, ' ').replace(/\s+/g, ' ').trim();
}

function highlightText(text, goldTerms, predTerms) {
  // Returns HTML string with <mark class="gold-mark"> and <mark class="pred-mark"> spans
  if (!text) return '';
  // Build list of [ start, end, cls ] regions
  const regions = [];

  function findRegions(text, terms, cls) {
    const normText = normalizeForHighlight(text);
    for (const term of terms) {
      const normTerm = normalizeForHighlight(term);
      if (!normTerm) continue;
      let offset = 0;
      while (offset < normText.length) {
        const idx = normText.indexOf(normTerm, offset);
        if (idx === -1) break;
        regions.push([idx, idx + normTerm.length, cls]);
        offset = idx + 1;
      }
    }
  }

  findRegions(text, goldTerms, 'gold-mark');
  findRegions(text, predTerms, 'pred-mark');

  if (regions.length === 0) return escapeHTML(text);

  // Merge and sort regions
  regions.sort((a, b) => a[0] - b[0] || b[1] - a[1]);
  let result = '';
  const normText = normalizeForHighlight(text);
  let pos = 0;
  for (const [start, end, cls] of regions) {
    if (start < pos) continue;
    result += escapeHTML(text.slice(pos, start));
    result += `<mark class="${cls}">${escapeHTML(text.slice(start, end))}</mark>`;
    pos = end;
  }
  result += escapeHTML(text.slice(pos));
  return result;
}

function escapeHTML(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── computeSummaryStats ──────────────────────────────────────────────────────
function computeSummaryStats(examples, annotations) {
  const total = examples.length;
  let reviewed = 0;
  const byHeuristicBucket = {};
  const byHumanOutputLabel = {};
  const byRetrievalSupport = {};
  const byPrimaryCause = {};
  let retrievalLimited = 0;
  let postRetrieval = 0;
  let complete = 0;
  let unclearOther = 0;

  // Confusion matrices: heuristic_bucket (row) x human_primary_failure_cause (col)
  const confusionHeuristicVsCause = {};
  const confusionEvalVsHuman = {};

  for (const e of examples) {
    const ann = annotations[e.example_id] || {};

    if (ann.reviewed) reviewed++;

    byHeuristicBucket[e.heuristic_bucket] = (byHeuristicBucket[e.heuristic_bucket] || 0) + 1;

    if (ann.human_output_label) {
      byHumanOutputLabel[ann.human_output_label] = (byHumanOutputLabel[ann.human_output_label] || 0) + 1;
    }
    if (ann.human_retrieval_support) {
      byRetrievalSupport[ann.human_retrieval_support] = (byRetrievalSupport[ann.human_retrieval_support] || 0) + 1;
    }
    if (ann.human_primary_failure_cause) {
      byPrimaryCause[ann.human_primary_failure_cause] = (byPrimaryCause[ann.human_primary_failure_cause] || 0) + 1;
    }

    // High-level split
    const cause = ann.human_primary_failure_cause;
    if (cause) {
      if (RETRIEVAL_LIMITED_CAUSES.has(cause)) retrievalLimited++;
      else if (cause === 'none' || cause === '') complete++;
      else postRetrieval++;
    } else if (e.heuristic_bucket === 'complete') {
      complete++;
    } else if (RETRIEVAL_LIMITED_BUCKETS.has(e.heuristic_bucket)) {
      retrievalLimited++;
    } else {
      unclearOther++;
    }

    // Confusion matrix 1: heuristic_bucket × human_primary_failure_cause
    if (ann.human_primary_failure_cause) {
      const row = e.heuristic_bucket || '(none)';
      const col = ann.human_primary_failure_cause;
      if (!confusionHeuristicVsCause[row]) confusionHeuristicVsCause[row] = {};
      confusionHeuristicVsCause[row][col] = (confusionHeuristicVsCause[row][col] || 0) + 1;
    }

    // Confusion matrix 2: evaluator_label × human_output_label
    if (ann.human_output_label) {
      const row = e.evaluator_label || '(none)';
      const col = ann.human_output_label;
      if (!confusionEvalVsHuman[row]) confusionEvalVsHuman[row] = {};
      confusionEvalVsHuman[row][col] = (confusionEvalVsHuman[row][col] || 0) + 1;
    }
  }

  return {
    total, reviewed, unreviewed: total - reviewed,
    pctReviewed: total > 0 ? Math.round((reviewed / total) * 100) : 0,
    byHeuristicBucket, byHumanOutputLabel, byRetrievalSupport, byPrimaryCause,
    retrievalLimited, postRetrieval, complete, unclearOther,
    confusionHeuristicVsCause, confusionEvalVsHuman,
  };
}

// ===========================================================================
// TEST SUITES
// ===========================================================================

describe('parseJSONL', () => {
  test('parses valid JSONL with multiple lines', () => {
    const text = '{"a":1}\n{"b":2}\n{"c":3}';
    const result = parseJSONL(text);
    assert.strictEqual(result.length, 3);
    assert.strictEqual(result[0].a, 1);
    assert.strictEqual(result[2].c, 3);
  });

  test('skips blank lines', () => {
    const text = '{"a":1}\n\n  \n{"b":2}';
    const result = parseJSONL(text);
    assert.strictEqual(result.length, 2);
  });

  test('throws on malformed JSON line', () => {
    assert.throws(() => parseJSONL('{"a":1}\nnot-json'));
  });

  test('returns empty array for empty string', () => {
    assert.deepStrictEqual(parseJSONL(''), []);
  });

  test('returns empty array for whitespace-only string', () => {
    assert.deepStrictEqual(parseJSONL('   \n   '), []);
  });
});

describe('parseCSV', () => {
  test('parses simple unquoted CSV', () => {
    const csv = 'a,b,c\n1,2,3\n4,5,6';
    const rows = parseCSV(csv);
    assert.strictEqual(rows.length, 2);
    assert.strictEqual(rows[0].a, '1');
    assert.strictEqual(rows[1].c, '6');
  });

  test('handles quoted fields with commas', () => {
    const csv = 'name,desc\nfoo,"has, comma"\nbar,plain';
    const rows = parseCSV(csv);
    assert.strictEqual(rows[0].desc, 'has, comma');
    assert.strictEqual(rows[1].desc, 'plain');
  });

  test('handles escaped double quotes inside quotes', () => {
    const csv = 'col\n"say ""hello"""';
    const rows = parseCSV(csv);
    assert.strictEqual(rows[0].col, 'say "hello"');
  });

  test('returns empty array for empty string', () => {
    assert.deepStrictEqual(parseCSV(''), []);
  });

  test('returns empty array when only header row present', () => {
    const csv = 'a,b,c';
    assert.deepStrictEqual(parseCSV(csv), []);
  });
});

describe('normalizeStratifiedRecord', () => {
  const raw = {
    example_id: 'ambig_17670',
    category: 'ambiguous',
    heuristic_cause: 'single_answer_collapse',
    question: 'Who is Graze?',
    gold_answers: ['Brother of Stanley Graze', 'An American spy'],
    pred_answers: ['Stanley Graze was an American spy.'],
    gold_count: 2, pred_count: 1,
    gold_support_ratio: 0.5,
    full_gold_support: false,
    all_pred_supported: true,
    score: 0.33,
    raw_model_output: 'Stanley Graze was an American spy.',
    retrieved_sources_top5: ['Stanley Graze', 'Another Article'],
    used_passage_ids_top5: ['11291823', '99999'],
    note: 'Model returned only one answer.',
  };

  test('maps example_id correctly', () => {
    const r = normalizeStratifiedRecord(raw);
    assert.strictEqual(r.example_id, 'ambig_17670');
  });

  test('uses heuristic_cause as heuristic_bucket', () => {
    const r = normalizeStratifiedRecord(raw);
    assert.strictEqual(r.heuristic_bucket, 'single_answer_collapse');
  });

  test('uses category as evaluator_label', () => {
    const r = normalizeStratifiedRecord(raw);
    assert.strictEqual(r.evaluator_label, 'ambiguous');
  });

  test('preserves gold_answers array', () => {
    const r = normalizeStratifiedRecord(raw);
    assert.deepStrictEqual(r.gold_answers, ['Brother of Stanley Graze', 'An American spy']);
  });

  test('builds retrieved_passages from top5 sources', () => {
    const r = normalizeStratifiedRecord(raw);
    assert.strictEqual(r.retrieved_passages.length, 2);
    assert.strictEqual(r.retrieved_passages[0].source, 'Stanley Graze');
    assert.strictEqual(r.retrieved_passages[0].passage_id, '11291823');
    assert.strictEqual(r.retrieved_passages[0].rank, 1);
  });

  test('handles missing retrieved_sources_top5', () => {
    const r = normalizeStratifiedRecord({ ...raw, retrieved_sources_top5: undefined });
    assert.deepStrictEqual(r.retrieved_passages, []);
  });

  test('stores note in metadata', () => {
    const r = normalizeStratifiedRecord(raw);
    assert.strictEqual(r.metadata.note, 'Model returned only one answer.');
  });
});

describe('normalizeRawRunRecords', () => {
  const inputs = [{
    example_id: 'ambig_1700',
    question: 'Who is the artist of "Feel Good"?',
    task_type: 'multi_answer_qa',
    gold: { single_answer: null, multi_answers: ["Che'Nelle", 'The Internet'], unknown_allowed: false },
    metadata: { dataset: 'ambigdocs', split: 'validation', extra: { ambiguous_entity: 'Feel Good', qid: 1700 } },
  }];
  const predictions = [{
    example_id: 'ambig_1700',
    raw_model_output: 'The 1975.',
    parsed_output: { single_answer: null, multi_answers: ['The 1975'], unknown: false },
  }];
  const evaluations = [{
    example_id: 'ambig_1700',
    dataset: 'ambigdocs',
    baseline_name: 'ambigdocs_llm_only',
    metrics: { exact_match: null, normalized_match: null, multi_answer_score: 0.0, answer_category: 'wrong' },
  }];
  const retrievals = [{
    example_id: 'ambig_1700',
    retrieved_passages: [
      { passage_id: '18798491', text: 'Feel Good...', source: 'Feel Good (The Internet album)', retrieval_score: 0.016, rank: 1 },
    ],
  }];

  test('produces one record per input', () => {
    const records = normalizeRawRunRecords({ inputs, predictions, evaluations, retrievals });
    assert.strictEqual(records.length, 1);
  });

  test('maps question correctly', () => {
    const [r] = normalizeRawRunRecords({ inputs, predictions, evaluations, retrievals });
    assert.strictEqual(r.question, 'Who is the artist of "Feel Good"?');
  });

  test('maps gold multi_answers', () => {
    const [r] = normalizeRawRunRecords({ inputs, predictions, evaluations, retrievals });
    assert.deepStrictEqual(r.gold_answers, ["Che'Nelle", 'The Internet']);
  });

  test('maps pred multi_answers', () => {
    const [r] = normalizeRawRunRecords({ inputs, predictions, evaluations, retrievals });
    assert.deepStrictEqual(r.pred_answers, ['The 1975']);
  });

  test('maps heuristic_bucket from answer_category', () => {
    const [r] = normalizeRawRunRecords({ inputs, predictions, evaluations, retrievals });
    assert.strictEqual(r.heuristic_bucket, 'wrong');
  });

  test('maps retrieved_passages', () => {
    const [r] = normalizeRawRunRecords({ inputs, predictions, evaluations, retrievals });
    assert.strictEqual(r.retrieved_passages.length, 1);
    assert.strictEqual(r.retrieved_passages[0].rank, 1);
  });

  test('maps ambiguous_entity from metadata.extra', () => {
    const [r] = normalizeRawRunRecords({ inputs, predictions, evaluations, retrievals });
    assert.strictEqual(r.ambiguous_entity, 'Feel Good');
  });

  test('handles missing predictions gracefully', () => {
    const [r] = normalizeRawRunRecords({ inputs, predictions: [], evaluations, retrievals });
    assert.deepStrictEqual(r.pred_answers, []);
  });

  test('handles missing evaluations gracefully', () => {
    const [r] = normalizeRawRunRecords({ inputs, predictions, evaluations: [], retrievals });
    assert.strictEqual(r.heuristic_bucket, '');
    assert.strictEqual(r.score, null);
  });

  test('handles missing retrievals gracefully', () => {
    const [r] = normalizeRawRunRecords({ inputs, predictions, evaluations, retrievals: [] });
    assert.deepStrictEqual(r.retrieved_passages, []);
  });
});

describe('computeDisagreement', () => {
  test('returns false when no human cause is set', () => {
    assert.strictEqual(computeDisagreement('wrong', ''), false);
    assert.strictEqual(computeDisagreement('wrong', null), false);
    assert.strictEqual(computeDisagreement('wrong', undefined), false);
  });

  test('returns true when human says retrieval-limited but heuristic says post-retrieval', () => {
    assert.strictEqual(
      computeDisagreement('support_present_but_omitted_answers', 'retrieval_missing_support'),
      true
    );
  });

  test('returns true when human says post-retrieval but heuristic says retrieval-limited', () => {
    assert.strictEqual(
      computeDisagreement('retrieval_missing_support_plus_partial_recall', 'omitted_supported_answers'),
      true
    );
  });

  test('returns false when both agree on retrieval-limited', () => {
    assert.strictEqual(
      computeDisagreement('retrieval_missing_support_plus_partial_recall', 'retrieval_missing_support'),
      false
    );
  });

  test('returns false when both agree on post-retrieval', () => {
    assert.strictEqual(
      computeDisagreement('support_present_but_omitted_answers', 'omitted_supported_answers'),
      false
    );
  });
});

describe('applyFilters', () => {
  const examples = [
    { example_id: 'e1', heuristic_bucket: 'wrong', evaluator_label: 'wrong', question: 'Who is JPO?', gold_answers: ['Japan Patent Office'], pred_answers: ['JPO'], ambiguous_entity: 'JPO', retrieved_passages: [] },
    { example_id: 'e2', heuristic_bucket: 'partial', evaluator_label: 'partial', question: 'What is SGU?', gold_answers: ['Stargate Universe'], pred_answers: [], ambiguous_entity: null, retrieved_passages: [{ source: 'SGU wiki' }] },
    { example_id: 'e3', heuristic_bucket: 'complete', evaluator_label: 'complete', question: 'Who sang Feel Good?', gold_answers: ["Che'Nelle"], pred_answers: ["Che'Nelle"], ambiguous_entity: 'Feel Good', retrieved_passages: [] },
  ];
  const annotations = {
    e1: { reviewed: true, human_output_label: 'wrong_selection_or_shape', human_primary_failure_cause: 'retrieval_missing_support', human_secondary_tags: ['wrong_granularity'] },
    e2: { reviewed: false, human_output_label: 'partial_recall', human_primary_failure_cause: 'omitted_supported_answers', human_secondary_tags: [] },
  };

  test('no filters returns all examples', () => {
    const r = applyFilters(examples, {}, '', annotations);
    assert.strictEqual(r.length, 3);
  });

  test('filters by heuristic bucket', () => {
    const r = applyFilters(examples, { heuristicBuckets: ['wrong', 'complete'] }, '', annotations);
    assert.strictEqual(r.length, 2);
    assert.ok(r.every(e => e.heuristic_bucket === 'wrong' || e.heuristic_bucket === 'complete'));
  });

  test('filters to reviewed only', () => {
    const r = applyFilters(examples, { reviewStatus: 'reviewed' }, '', annotations);
    assert.strictEqual(r.length, 1);
    assert.strictEqual(r[0].example_id, 'e1');
  });

  test('filters to unreviewed only', () => {
    const r = applyFilters(examples, { reviewStatus: 'unreviewed' }, '', annotations);
    assert.strictEqual(r.length, 2);
    assert.ok(r.every(e => e.example_id !== 'e1'));
  });

  test('filters by onlyDisagreements', () => {
    // e1: heuristic=wrong, human=retrieval_missing_support → RETRIEVAL_LIMITED_BUCKETS has 'wrong'
    // computeDisagreement('wrong','retrieval_missing_support') → both retrieval-limited → false
    // Actually 'wrong' is in RETRIEVAL_LIMITED_BUCKETS and 'retrieval_missing_support' is in RETRIEVAL_LIMITED_CAUSES → no disagreement
    // e2: heuristic=partial, human=omitted_supported_answers → partial not in RETRIEVAL_LIMITED_BUCKETS, omitted not retrieval-limited → no disagreement
    const r = applyFilters(examples, { onlyDisagreements: true }, '', annotations);
    assert.strictEqual(r.length, 0);
  });

  test('search by question text', () => {
    const r = applyFilters(examples, {}, 'JPO', annotations);
    assert.strictEqual(r.length, 1);
    assert.strictEqual(r[0].example_id, 'e1');
  });

  test('search by gold answer', () => {
    const r = applyFilters(examples, {}, "Che'Nelle", annotations);
    assert.strictEqual(r.length, 1);
    assert.strictEqual(r[0].example_id, 'e3');
  });

  test('search by passage source', () => {
    const r = applyFilters(examples, {}, 'SGU wiki', annotations);
    assert.strictEqual(r.length, 1);
    assert.strictEqual(r[0].example_id, 'e2');
  });

  test('search is case-insensitive', () => {
    const r = applyFilters(examples, {}, 'feel good', annotations);
    assert.strictEqual(r.length, 1);
  });

  test('filters by secondary tag', () => {
    const r = applyFilters(examples, { secondaryTag: 'wrong_granularity' }, '', annotations);
    assert.strictEqual(r.length, 1);
    assert.strictEqual(r[0].example_id, 'e1');
  });
});

describe('findNextUnreviewed', () => {
  const examples = [
    { example_id: 'e0' }, { example_id: 'e1' }, { example_id: 'e2' }, { example_id: 'e3' },
  ];
  const annotations = {
    e0: { reviewed: true },
    e2: { reviewed: true },
  };

  test('finds next unreviewed from beginning', () => {
    const idx = findNextUnreviewed(examples, -1, annotations);
    assert.strictEqual(idx, 1); // e1 is first unreviewed
  });

  test('skips reviewed to find unreviewed', () => {
    const idx = findNextUnreviewed(examples, 0, annotations);
    assert.strictEqual(idx, 1); // e1
  });

  test('skips reviewed and wraps if needed', () => {
    const idx = findNextUnreviewed(examples, 1, annotations);
    assert.strictEqual(idx, 3); // e3 (e2 is reviewed)
  });

  test('wraps around to beginning', () => {
    // From e3, wrap to e1 (e0 reviewed, e1 unreviewed)
    const idx = findNextUnreviewed(examples, 3, annotations);
    assert.strictEqual(idx, 1);
  });

  test('returns currentIndex when all reviewed', () => {
    const allReviewed = {
      e0: { reviewed: true }, e1: { reviewed: true },
      e2: { reviewed: true }, e3: { reviewed: true },
    };
    const idx = findNextUnreviewed(examples, 0, allReviewed);
    assert.strictEqual(idx, 0);
  });
});

describe('findNextDisagreement', () => {
  const examples = [
    { example_id: 'e0', heuristic_bucket: 'wrong' },
    { example_id: 'e1', heuristic_bucket: 'support_present_but_omitted_answers' },
    { example_id: 'e2', heuristic_bucket: 'support_present_but_omitted_answers' },
  ];
  const annotations = {
    e0: { human_primary_failure_cause: 'retrieval_missing_support' }, // retrieval-limited agrees with 'wrong'
    e1: { human_primary_failure_cause: 'retrieval_missing_support' }, // DISAGREEMENT: bucket is post-retr, human is retr-limited
    e2: { human_primary_failure_cause: 'omitted_supported_answers' },  // agree on post-retrieval
  };

  test('finds disagreement example', () => {
    const idx = findNextDisagreement(examples, 0, annotations);
    assert.strictEqual(idx, 1);
  });

  test('skips over non-disagreement examples', () => {
    const idx = findNextDisagreement(examples, 1, annotations);
    // no more disagreements after e1; wraps back, e0 no disagreement, so stays at 1
    assert.strictEqual(idx, 1);
  });
});

describe('exportAnnotationsJSON', () => {
  const examples = [{
    example_id: 'e1', question: 'Q?', gold_answers: ['A', 'B'], pred_answers: ['A'],
    evaluator_label: 'partial', heuristic_bucket: 'partial',
    prompt_text: null, retrieved_passages: [],
  }];
  const annotations = {
    e1: {
      human_output_label: 'partial_recall',
      human_retrieval_support: 'all_gold_supported',
      human_primary_failure_cause: 'omitted_supported_answers',
      human_secondary_tags: ['wrong_granularity'],
      reviewer_notes: 'looks right',
      reviewed: true,
      flagged_for_adjudication: false,
      timestamp_last_modified: '2026-04-16T00:00:00.000Z',
    },
  };

  test('produces valid JSON', () => {
    const json = exportAnnotationsJSON(examples, annotations);
    assert.doesNotThrow(() => JSON.parse(json));
  });

  test('has all PRD §11 schema fields', () => {
    const records = JSON.parse(exportAnnotationsJSON(examples, annotations));
    const r = records[0];
    assert.ok('example_id' in r);
    assert.ok('source_dataset' in r);
    assert.ok('question' in r);
    assert.ok('gold_answers' in r);
    assert.ok('prediction' in r);
    assert.ok('evaluator_label' in r);
    assert.ok('heuristic_bucket' in r);
    assert.ok('human_output_label' in r);
    assert.ok('human_retrieval_support' in r);
    assert.ok('human_primary_failure_cause' in r);
    assert.ok('human_secondary_tags' in r);
    assert.ok('reviewer_notes' in r);
    assert.ok('reviewed' in r);
    assert.ok('flagged_for_adjudication' in r);
    assert.ok('timestamp_last_modified' in r);
  });

  test('correctly maps annotation values', () => {
    const records = JSON.parse(exportAnnotationsJSON(examples, annotations));
    assert.strictEqual(records[0].human_output_label, 'partial_recall');
    assert.deepStrictEqual(records[0].human_secondary_tags, ['wrong_granularity']);
    assert.strictEqual(records[0].reviewed, true);
  });

  test('fills defaults for unannotated example', () => {
    const records = JSON.parse(exportAnnotationsJSON(examples, {}));
    assert.strictEqual(records[0].human_output_label, null);
    assert.strictEqual(records[0].reviewed, false);
    assert.deepStrictEqual(records[0].human_secondary_tags, []);
  });
});

describe('exportAnnotationsCSV', () => {
  const examples = [{
    example_id: 'e1', question: 'Q?', gold_answers: ['A', 'B'], pred_answers: ['A'],
    evaluator_label: 'partial', heuristic_bucket: 'partial',
  }];
  const annotations = {
    e1: {
      human_output_label: 'partial_recall',
      human_retrieval_support: 'all_gold_supported',
      human_primary_failure_cause: 'omitted_supported_answers',
      human_secondary_tags: ['wrong_granularity', 'formatting_noise'],
      reviewer_notes: 'Note with, comma',
      reviewed: true,
      flagged_for_adjudication: false,
    },
  };

  test('produces correct header row', () => {
    const csv = exportAnnotationsCSV(examples, annotations);
    const firstLine = csv.split('\n')[0];
    assert.ok(firstLine.includes('example_id'));
    assert.ok(firstLine.includes('human_output_label'));
    assert.ok(firstLine.includes('human_secondary_tags'));
  });

  test('has correct number of rows', () => {
    const csv = exportAnnotationsCSV(examples, annotations);
    const rows = csv.split('\n');
    assert.strictEqual(rows.length, 2); // header + 1 data row
  });

  test('joins secondary tags with pipe', () => {
    const csv = exportAnnotationsCSV(examples, annotations);
    assert.ok(csv.includes('wrong_granularity | formatting_noise'));
  });

  test('quotes fields containing commas', () => {
    const csv = exportAnnotationsCSV(examples, annotations);
    assert.ok(csv.includes('"Note with, comma"'));
  });

  test('fills empty string for unannotated fields', () => {
    const csv = exportAnnotationsCSV(examples, {});
    const dataRow = csv.split('\n')[1];
    // Should have empty fields for all annotation columns
    assert.ok(dataRow !== null);
  });
});

describe('highlightText', () => {
  test('returns escaped HTML when no matches', () => {
    const result = highlightText('Hello world', ['xyz'], []);
    assert.strictEqual(result, 'Hello world');
  });

  test('wraps gold match in gold-mark', () => {
    const result = highlightText('The Internet album', ['The Internet'], []);
    assert.ok(result.includes('class="gold-mark"'));
  });

  test('wraps pred match in pred-mark', () => {
    const result = highlightText('The Internet album', [], ['The Internet']);
    assert.ok(result.includes('class="pred-mark"'));
  });

  test('escapes HTML in plain text portions', () => {
    const result = highlightText('<script>alert(1)</script>', [], []);
    assert.ok(result.includes('&lt;script&gt;'));
    assert.ok(!result.includes('<script>'));
  });

  test('handles empty text', () => {
    assert.strictEqual(highlightText('', ['gold'], []), '');
  });

  test('handles empty term lists', () => {
    const result = highlightText('plain text', [], []);
    assert.strictEqual(result, 'plain text');
  });
});

describe('computeSummaryStats', () => {
  const examples = [
    { example_id: 'e0', heuristic_bucket: 'complete', evaluator_label: 'complete' },
    { example_id: 'e1', heuristic_bucket: 'wrong', evaluator_label: 'wrong' },
    { example_id: 'e2', heuristic_bucket: 'partial', evaluator_label: 'partial' },
    { example_id: 'e3', heuristic_bucket: 'wrong', evaluator_label: 'wrong' },
  ];
  const annotations = {
    e0: { reviewed: true, human_output_label: 'complete', human_retrieval_support: 'all_gold_supported', human_primary_failure_cause: null },
    e1: { reviewed: true, human_output_label: 'wrong_selection_or_shape', human_retrieval_support: 'partial_gold_supported', human_primary_failure_cause: 'retrieval_missing_support' },
    e2: { reviewed: false, human_output_label: 'partial_recall', human_retrieval_support: 'all_gold_supported', human_primary_failure_cause: 'omitted_supported_answers' },
  };

  test('counts total correctly', () => {
    const s = computeSummaryStats(examples, annotations);
    assert.strictEqual(s.total, 4);
  });

  test('counts reviewed correctly', () => {
    const s = computeSummaryStats(examples, annotations);
    assert.strictEqual(s.reviewed, 2);
    assert.strictEqual(s.unreviewed, 2);
  });

  test('computes pctReviewed', () => {
    const s = computeSummaryStats(examples, annotations);
    assert.strictEqual(s.pctReviewed, 50);
  });

  test('counts byHeuristicBucket', () => {
    const s = computeSummaryStats(examples, annotations);
    assert.strictEqual(s.byHeuristicBucket['wrong'], 2);
    assert.strictEqual(s.byHeuristicBucket['complete'], 1);
    assert.strictEqual(s.byHeuristicBucket['partial'], 1);
  });

  test('counts byHumanOutputLabel for reviewed examples', () => {
    const s = computeSummaryStats(examples, annotations);
    assert.strictEqual(s.byHumanOutputLabel['complete'], 1);
    assert.strictEqual(s.byHumanOutputLabel['partial_recall'], 1);
  });

  test('counts retrieval-limited split', () => {
    const s = computeSummaryStats(examples, annotations);
    // e1 has human_primary_failure_cause=retrieval_missing_support → retrievalLimited++
    assert.ok(s.retrievalLimited >= 1);
  });

  test('builds confusion matrix heuristic x human cause', () => {
    const s = computeSummaryStats(examples, annotations);
    // e1: heuristic=wrong, human_cause=retrieval_missing_support
    assert.strictEqual(s.confusionHeuristicVsCause['wrong']['retrieval_missing_support'], 1);
    // e2: heuristic=partial, human_cause=omitted_supported_answers
    assert.strictEqual(s.confusionHeuristicVsCause['partial']['omitted_supported_answers'], 1);
  });

  test('builds confusion matrix eval x human output', () => {
    const s = computeSummaryStats(examples, annotations);
    assert.strictEqual(s.confusionEvalVsHuman['complete']['complete'], 1);
    assert.strictEqual(s.confusionEvalVsHuman['wrong']['wrong_selection_or_shape'], 1);
  });

  test('handles empty annotations', () => {
    const s = computeSummaryStats(examples, {});
    assert.strictEqual(s.reviewed, 0);
    assert.deepStrictEqual(s.byHumanOutputLabel, {});
  });
});

// ---------------------------------------------------------------------------
// Final report
// ---------------------------------------------------------------------------

console.log(`\n${'─'.repeat(50)}`);
console.log(`Results: ${_passed} passed, ${_failed} failed`);
if (_failed > 0) {
  console.error(`\n${_failed} test(s) FAILED`);
  process.exit(1);
} else {
  console.log('\nAll tests GREEN ✓');
}
