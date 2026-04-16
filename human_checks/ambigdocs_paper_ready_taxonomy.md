# Paper-Ready Error Taxonomy for the Uploaded AmbigDocs Run

## Scope

This taxonomy is designed for the uploaded AmbigDocs validation run using hybrid retrieval, reranking, Qwen2.5-32B-Instruct, multi-answer prompting, and full-context prompting.

It extends the original AmbigDocs output ontology by adding **causal buckets** that separate retrieval-stage failures from post-retrieval answer construction failures.

Use the counts below as a **run-specific empirical summary**, not as a universal AmbigDocs distribution.


## Summary table

| Error type | Definition | Count | % | Canonical example(s) | Interpretation |
|---|---|---:|---:|---|---|
| Retrieval missing support | The retrieved set appears not to contain support for the full gold answer set, so even a perfect generator could not fully solve the example from the available evidence. | 1065 | 29.5 | Q: "What does JPO refer to?" Gold: [Japan Patent Office; Johannesburg Philharmonic Orchestra]. Pred: ["Japan Patent Office"].<br><br>Q: "What does SGU refer to?" Gold has 5 valid expansions. Pred returns only [Swiss German University; Stargate Universe], consistent with partial retrieval support. | Mainly a retrieval-stage failure. Prioritize better ambiguity-aware recall, denser entity coverage, or explicit entity expansion before generation. |
| Supported but omitted valid answers | All gold answers appear to be supported somewhere in the retrieved set, but the model lists only a subset of them. | 825 | 22.9 | Q: "Where was Expo '29 held?" Gold: [Seville, Spain; Newcastle, Tyne and Wear, England; 1929 Barcelona International Exposition]. Pred: [Newcastle, Tyne and Wear; Seville, Spain].<br><br>Q: "What is the profession of Yang Dan?" Gold: [Chemist and chemical biologist; Neuroscientist; Author/co-author of over 20 patents...]. Pred: [Chemist and chemical biologist; Neuroscientist]. | Post-retrieval aggregation failure. Evidence exists, but the model does not complete the answer set. |
| Single-answer collapse | The model treats an inherently multi-answer ambiguity case as ordinary single-answer QA and returns only one referent or one answer. | 478 | 13.2 | Q: "When was David Simpkins born?" Gold: [29 May 1934; 1962]. Pred: [29 May 1934].<br><br>Q: "Who is Graze?" Gold contains 3 valid referents. Pred returns only one referent: [American spy and economist from New York City]. | Answer-mode failure. The model is not reliably adopting the multi-answer behavior requested by the prompt. |
| Supported but wrong selection/shape | Relevant evidence is present, but the model still fails because it chooses the wrong granularity, wrong referent, wrong framing, or an evaluator-sensitive answer form. | 534 | 14.8 | Q: "What is Nasrullah?" Gold: [A 16th-century stone arch bridge; A 16th-century Ottoman mosque in Kastamonu, Turkey]. Pred: [Nasrullah Mosque; Nasrullah Bridge].<br><br>Q: "What does PIAS stand for?" Gold requires both the company description and the label origin. Pred: [Play It Again Sam]. | Not a retrieval miss. The model is reading the right evidence but outputting the wrong abstraction level or answer form. |
| Supported but unsupported generation | Gold-supporting evidence is present, but the model adds unsupported details, drops supported specifics, or answers in a way that is not actually licensed by the retrieved text. | 137 | 3.8 | Q: "When was Jim Stevenson born?" Gold: [1935 in Falkirk; 17 May 1992 in Luton, Bedfordshire]. Pred: [17 May 1992; 1935 (exact date not specified)].<br><br>Q: "When was Stuart Elliott born?" Gold: [27 August 1977; 23 July 1978; 22 May 1953]. Pred: [1953; 1977; 1978]. | Reasoning/normalization failure with evidence present. Candidate fixes include answer extraction constraints or evidence-linked decoding. |
| Entity merge | The model blends multiple referents, senses, or answer types into a mixed response instead of cleanly separating them. | 26 | 0.7 | Q: "What is \"Crazy Rich Asians\"?" Gold separates novel and film senses. Pred gives two mixed descriptions, including a film answer that is explicitly tied back to the novel.<br><br>Q: "What is Ethelton?" Gold: [A suburb; A railway station]. Pred adds both plus unrelated district information, blending referents into one response. | Classic disambiguation failure. Rare here, but conceptually important because it indicates cross-entity contamination rather than simple omission. |
| Complete (reference row) | The model returns all valid answers required by the example. | 545 | 15.1 | Q: "Who is the artist of the album \"Feel Good\"?" Gold: [Che'Nelle; The Internet]. Pred: [The Internet; Che'Nelle]. | Useful as a reference row when comparing failure modes against a successful multi-answer baseline. |

## Recommended framing for the paper

A clean way to present this taxonomy is as a two-stage decomposition:

1. **Is the full gold answer set supported by retrieval?**

2. **If yes, does the model still fail to aggregate, separate, or normalize the supported answers correctly?**


That yields a principled split between:

- **retrieval-limited failures**: retrieval missing support

- **post-retrieval failures**: omitted supported answers, single-answer collapse, wrong selection/shape, unsupported generation, entity merge


## Suggested taxonomy names for a paper figure or table

- Retrieval missing support

- Omitted supported answers

- Single-answer collapse

- Wrong selection / wrong answer shape

- Unsupported generation despite evidence

- Entity merge


## One-sentence takeaway

The dominant error profile in this run is **not generic hallucination**. It is mostly a mixture of **retrieval incompleteness** and **post-retrieval multi-answer aggregation failures**, with a smaller but important set of **entity-resolution failures**.
