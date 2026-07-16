# Discussed but Not Yet Implemented Changes

Changes discussed during development that have not been implemented yet.
Mark as done with `**Done YYYY-MM-DD**` when implemented.

---

## Performance

### P1 — Batch Neo4j writes in enricher (UNWIND)
**Done 2026-06-04**\
**File:** `kg/graph_enricher.py` → `_push_mapping()`\
**Expected impact:** Highest. Currently opens 3 separate Neo4j sessions per entity (MERGE OntologyTerm, MERGE DOC_MAPPED_TO edge, SET upheno_enriched) plus 2 more per IS_A ancestor. At ~10k entities with ~3 ancestors each this is ~40k individual round trips run serially. Fix: collect all mappings first, push in 2 UNWIND queries (one for all OntologyTerm nodes, one for all edges + flags). Same pattern already used in `graph_indexer.py`. Estimated impact: 10–50× fewer round trips, likely cuts enrichment from ~5h to ~15–30 min for 8 species.\
**Implemented:** Replaced per-entity `_push_mapping()` loop in `_run_upheno_mapping()` with 4 UNWIND queries (A: MERGE OntologyTerm nodes, B: MERGE DOC_MAPPED_TO + mark enriched, C: mark no-match, D: ancestor nodes + IS_A edges grouped by rel_type). Also deduplicates ancestors by term_id before owlready2 lookup.\

### P2 — Batch similarity edge writes (UNWIND)
**Done 2026-06-04**\
**File:** `kg/graph_enricher.py` → `_run_similarity_edges()`\
**Expected impact:** Medium. Currently writes one `session.run()` per passing pair inside the O(n²) loop. Fix: collect all pairs above threshold into a list, write as a single UNWIND query.\
**Implemented:** Collects all passing pairs into `pairs_to_write`, writes in chunks of 1000 via UNWIND.\

### P3 — Increase `--map-workers` in batch runs
**File:** `scripts/run_batch_tsv.py`, `pipeline/run_graph_pipeline.py`\
**Expected impact:** Medium. All batch runs so far used default of 1 worker. The trait mapper has Stage 2 (cosine, CPU-only) and cache-hit paths that don't touch the LLM at all and benefit immediately from parallelism. Even LLM-bound Stage 1/3 benefit if the server handles light concurrency. Needs testing to find the right value without overloading the server.\

### P5 — Batch Stage 3 LLM verification calls (verify_candidate)
**File:** `kg/trait_mapper.py` → `verify_candidate()` and `map_traits_batch()`\
**Expected impact:** High. Stage 3 (`verify_candidate`) is the primary enrichment bottleneck — one HTTP round trip per entity (~2–4s each). Stage 1 normalization is already batched (`norm_batch_size`). Stage 3 is not. Fix: collect N entities that reach Stage 3 (after cosine early-exit filtering), send all N (entity + candidate list) in a single LLM prompt, parse a JSON array of N decisions. Batch size of 5–10 would reduce round trips 5–10×, estimated impact 2–4× reduction in total Stage 3 time.\
**Risks:** (1) LLM attention dilution at large batch sizes — safe range is ~5, risky above ~15. (2) A malformed JSON response from the LLM loses the whole batch; needs fallback to individual calls on parse failure. (3) Parent broadening (`confidence: low` → re-verify with parent candidates) needs a two-pass design: batch-verify all entities first, then collect low-confidence ones for a second batch pass. (4) Token budget per entity shrinks — may need to reduce candidate definitions or top-k to keep prompt size manageable.\
**Notes:** P4 (pre-filter generic entities) is a prerequisite that reduces the entity population before Stage 3 and should be implemented first. Pilot at batch_size=5 with full fallback before increasing.\

### P3b — Hard-delete junk entities at indexing time
**Done 2026-07-05**\
**File:** `kg/graph_indexer.py` → `_is_junk_entity()` (line 239) + guard at line 320\
**What:** LLM entity extraction pulled out citations (`"Park et al. 2017"`), measurements (`"13–28 °C"`), accession IDs (`"XM_038812782.1"`), non-ASCII text, and author-year references alongside real biological entities. These junk entities created spurious `DOC_SIMILAR_TO` Tier 2 edges (e.g. two species sharing a citation → fake cross-species similarity signal).\
**Implemented:** `_is_junk_entity()` applies five rules before Neo4j MERGE — drops entities that: contain `et al`, match `^[A-Z][a-z]+ \d{4}$` (author-year), start with a digit (measurements), match accession ID patterns (`XM_`, `NM_`, `contig`), or contain non-ASCII characters. Entities starting with a letter that contain numbers mid-string are NOT dropped.

### P4 — Pre-filter generic entity surface forms before uPheno mapping
**File:** `kg/graph_enricher.py` → `_run_upheno_mapping()`\
**Expected impact:** Low-medium. Entities like `"cell"`, `"tissue"`, `"protein"`, `"gene"` are too generic to map to a useful specific uPheno term. They consume LLM calls and produce high-level useless ancestors that add noise to Tier 1. A blocklist or length/frequency filter applied before `map_traits_batch()` would skip them. Also improves Tier 1 signal quality.\

---

## Output Quality — KG Node Reranking

### Q1 — Ontology depth scoring for Tier 1 (early convergence preference)
**Done 2026-06-11**\
**File:** `kg/graph_synthesizer.py` → `_query_tier1()`\
**What:** Depth scoring (original proposal) was superseded by Q2 fan-in, which is a strictly better specificity signal. A term at depth 1 can still have high global fan-in (semi-generic); a term at depth 3 can have low fan-in (highly specific). The IC-style scoring in Q2 captures this correctly without needing depth tracking.\
**Implemented:** Replaced Cypher `ORDER BY size(species_list) DESC` with Python IC-style scoring via `_score_tier1_term()`. Depth traversal approach skipped in favour of Q2.

### Q2 — Global fan-in scoring for Tier 1 (ontology node specificity)
**Done 2026-06-11 | Option B integrated 2026-06-17**\
**Files:** `kg/precompute_fan_in.py` (new), `kg/neo4j_client.py`, `kg/graph_synthesizer.py`, `pipeline/run_graph_pipeline.py`\
**What:** Precomputes `global_fan_in` (number of distinct descendant OntologyTerm nodes reachable via IS_A*1..) on every OntologyTerm node in Neo4j. Used in `_query_tier1()` as an IC-style specificity signal to replace the crude `--tier1-max-entity-forms` hard cap.

**Root problem:** The old Cypher `ORDER BY size(species_list) DESC` always surfaced pan-biological terms (e.g. "abnormal anatomical structure", 8/8 species, fan_in ~8000) at position #1, while gene-relevant specific terms ("cerebellar hypoplasia", 4/8 species, fan_in ~5) ranked #18 or got cut by LIMIT 30. Every N=8 synthesis produced the same generic communities regardless of which gene was being studied.

**How the reranking works:**

1. **Precompute (offline):** `kg/precompute_fan_in.py` runs a single traversal query across all OntologyTerm IS_A edges:
   ```cypher
   MATCH (descendant:OntologyTerm)-[:IS_A*1..]->(ancestor:OntologyTerm)
   WITH ancestor, count(DISTINCT descendant) AS fan_in
   RETURN ancestor.term_id AS term_id, fan_in
   ```
   Results are written back onto each node as `global_fan_in` in 1000-node UNWIND chunks via `run_write_query()`. High fan_in = generic ("abnormal anatomical structure" ~8000); low fan_in = specific ("cerebellar hypoplasia" ~5).

2. **Cypher change:** `_query_tier1()` now fetches a larger pool (LIMIT 300, no ORDER BY) and includes `coalesce(ancestor.global_fan_in, -1) AS fan_in` in the return. `-1` means precompute hasn't been run yet.

3. **IC-style Python scoring** via `_score_tier1_term()`:
   ```
   breadth    = species_count / n_species           (0..1 — how many species share this term)
   ic         = log((max_fan_in + 1) / (fan_in + 1)) (higher = more specific in the ontology)
   ef_penalty = 1 / (1 + 0.05 × entity_forms)       (soft penalty for catch-all terms)
   score      = breadth × ic × ef_penalty
   ```
   Top 30 by score are kept (`MAX_TIER1_RESULTS = 30`).

4. **Graceful fallback:** If `fan_in == -1` for all candidates (precompute not run), a warning is printed and the old species-count sort is used instead — synthesis still works, just with generic rankings.

5. **Option B — automatic integration (2026-06-17):** `pipeline/run_graph_pipeline.py` now calls `precompute_fan_in.run()` automatically between enrichment and synthesis (when both are active):
   ```python
   if not args.skip_enrich and not args.skip_synthesis and species_norms:
       print("[GraphPipeline] Updating global_fan_in on OntologyTerm nodes...")
       try:
           from kg.precompute_fan_in import run as _run_fan_in
           _run_fan_in()
       except Exception as exc:
           print(f"[GraphPipeline] precompute_fan_in failed (non-fatal, synthesis will use fallback ranking): {exc}")
   ```
   No manual step required — the first full pipeline run writes `global_fan_in` before synthesis executes. Re-runs that use `--skip-enrich` still benefit because the values persist in Neo4j from the last full run.

**Implemented:**\
- `kg/precompute_fan_in.py` — offline script; also called automatically by the pipeline\
- `kg/neo4j_client.py` — added `run_write_query()` (write-capable counterpart to `run_query()`) and `get_driver` public alias\
- `kg/graph_synthesizer.py` — added `_score_tier1_term()`, modified `_query_tier1()`: Cypher returns `coalesce(ancestor.global_fan_in, -1) AS fan_in` with LIMIT 300 (no ORDER BY); Python ranks with IC scoring; graceful fallback to species-count sort if precompute hasn't been run\
- `pipeline/run_graph_pipeline.py` — raised `--tier1-max-entity-forms` default 50 → 200 (now a safety-net cap, not the primary specificity filter); added automatic `precompute_fan_in.run()` call between enrichment and synthesis\

**Ranking example (SPATC1L, N=8, max_fan_in ~8000):**

| Term | Species | Fan-in | Ent. forms | Old rank | IC score | New rank |
|---|---|---|---|---|---|---|
| abnormal anatomical structure | 8/8 | ~8,000 | 48 | #1 | 1.0 × log(1) × 0.51 ≈ **0.00** | **#28** |
| abnormal immune response | 8/8 | ~200 | 12 | #3 | 1.0 × log(40) × 0.70 ≈ **2.60** | #8 |
| cerebellar hypoplasia | 4/8 | ~5 | 2 | #18 | 0.5 × log(1600) × 0.91 ≈ **3.30** | **#2** |
| scleral tissue cyst | 3/8 | ~3 | 1 | filtered | 0.38 × log(2667) × 0.95 ≈ **2.96** | **#5** |

**Run order:** Full pipeline run (`run_graph_pipeline.py` without `--skip-enrich`) triggers precompute automatically. To rerun synthesis only: `python -m kg.precompute_fan_in` first, then `--skip-ingest --skip-enrich`.\

---

## Output Quality — Signal Specificity

### Q3 — Fix `similarity_pairs_added: 0` (Tier 2 DOC_SIMILAR_TO edges never created)
**File:** `kg/graph_enricher.py` → `_run_similarity_edges()`\
**What:** Zero DOC_SIMILAR_TO edges have been created across all runs. Root cause not yet investigated. Tier 2 is designed to bridge vocabulary fragmentation — the same concept expressed differently across species (e.g. "myoglobin" vs "breath-hold diving") would cluster via embedding similarity. Without it, the hypoxia signal in TC6 split into fragments that individually fell below `min_species`. Fixing this is prerequisite for meaningful `"merged"` tier communities.\
**Status:** Root cause unknown. Needs investigation before implementation.\
-> Lower Threshold, rest makes sense as implemented \


### Q4 — Species sub-clustering for large testcases (N > 7)
**File:** New module or extension to `pipeline/run_graph_pipeline.py`\
**What:** For testcases with N > 7 species, `min_species = N//2` becomes so strict that only pan-mammalian biology survives. Fix: split species into overlapping sub-groups of 3–5, run synthesis on each sub-group independently, then merge/deduplicate the resulting communities. Communities that appear in multiple sub-groups are promoted as high-confidence. This keeps `min_species` at a level where specific signal can survive while still covering all species.\
**Risk:** Medium-high. Requires new orchestration logic and a community deduplication/merging step. Needs careful design to avoid double-counting species or inflating community scores.\

---

## Performance — Indexing Parallelism

### P6 — Parallel chunk processing in graph_indexer.py
**File:** `kg/graph_indexer.py` → `index_species()` inner loop\
**Expected impact:** Highest. Indexing is 96-97% of total pipeline wall time (~330s per 1000 entities). Currently each chunk is processed synchronously: extract entities (LLM call) → MERGE into Neo4j → next chunk. With 655–1531 chunks per species, that is 655–1531 sequential LLM round trips. Parallelising at 4× concurrency would cut indexing to ~25% of current time, reducing TC4 from 1h 50m to ~28m and TC6 from 2h 42m to ~40m.\
**Design:** Worker pool over chunks; each worker calls the LLM and collects (chunk_id, entities) tuples; a single writer thread batches MERGE writes via UNWIND. Requires fcntl-style serialisation for the writer only, not the LLM calls.\
**Risk:** Medium. Need to handle LLM server concurrency limits. Start with 2–4 workers and confirm server stability before increasing.

### P7 — Cap OpenAlex Retry-After header
**File:** `rag_cli.py` or wherever OpenAlex synonym fetching happens\
**Expected impact:** Prevents catastrophic 23-hour stalls. One TC6 run waited 83,291s (23h) for a "Fin-backed Whale" synonym lookup. Any Retry-After > 120s should be treated as "skip this synonym" rather than "sleep and retry".\
**Implementation:** After receiving a 429 response, if `Retry-After > 120`: log a warning, skip the synonym, continue with remaining sources. If `Retry-After <= 120`: sleep and retry as now.

### P9 — Strip reference sections before chunking
**Done 2026-07-16**\
**File:** `core/rag_cli.py` → `_strip_references()` applied in `load_ocr()`, `ingest_pmc_texts()`, `ingest_wikipedia()`\
**What:** PDFs, PMC full texts, and Wikipedia articles all contain a References/Bibliography section at the end that has zero biological trait value. Chunking the reference list produces entries like "Smith et al. 2017. Journal of..." which the junk entity filter has to catch after the fact. Truncating at the reference header removes 10–25% of paper length before chunking, reducing chunk count and eliminating the source of et-al junk entities entirely.\
**Implementation:** `_REF_HEADER_RE` regex matches "References", "Bibliography", "Literature Cited", "Works Cited", "Acknowledgements" as section headers. Text is truncated at the first match. Logs the percentage removed per document.

### P10 — First/last 15% paper chunk cap
**Done 2026-07-16**\
**Files:** `core/rag_cli.py` → `_apply_chunk_cap()`, `RAG.__init__(paper_chunk_cap)`, `load_ocr()`, `ingest_pmc_texts()`; `pipeline/graph_inventory_single.py` → `--paper-chunk-cap`; `pipeline/run_graph_pipeline.py` → `--paper-chunk-cap`\
**What:** For papers (PDF + PMC only, not Wikipedia), keep only the first N% and last N% of chunks, dropping the middle. Hypothesis: abstract + introduction (first ~15%) and discussion/conclusion (last ~15%) contain most biological trait signal; methods + raw results (middle ~70%) contribute less signal per chunk. Reduces chunks per paper from ~65 to ~20 (~3× speedup on indexing for paper chunks). Wikipedia is excluded because its structure is different (traits are in middle sections, not just intro/outro).\
**Scripts:** `scripts/run_testcases_chunkcap.sh` uses `--paper-chunk-cap 0.15`; `scripts/run_testcases_logging.sh` runs without the cap for comparison.\
**Status:** Implemented. Quality impact vs full run not yet measured — run both scripts to compare synthesis output.

### P8 — Validate species names before ingest (genus vs. binomial)
**File:** ingestion layer, before Wikipedia/PMC fetching\
**Expected impact:** Medium. TC6 phyllotis ingest took 1039s (5.7× average) because phyllotis is a genus name, triggering disambiguation across all member species. A GBIF or simple heuristic check (does the name contain exactly one space? does it resolve to a single species?) before ingest would catch this at <1s and warn/skip rather than spending 17 minutes fetching genus-wide documents.\
**Implementation:** Quick GBIF species match as already used in `build_testcase_json.py`; if match returns >1 canonical result or is genus-rank, warn and either abort or use the first canonical match.

---

## Robustness

### R1 — Fix LLM malformed JSON crash in graph_indexer.py
**File:** `kg/graph_indexer.py` → `_extract_entities_from_chunk()`\
**What:** TC5 crashed after 6/7 species indexed (~5.5h of runtime lost) because Qwen 3.5:27b returned extra content after a valid JSON object (`Extra data: line 18 column 1 (char 500)`). Currently the crash propagates and kills the entire run.\
**Fix:** Wrap JSON parsing in a try/except; on parse failure, attempt to extract the JSON substring up to the first `]` or `}` boundary; if that also fails, log the raw LLM output and return an empty entity list for that chunk (skip-chunk fallback). No entities are better than a crash that loses all subsequent species.

### R2 — Fix Chroma SQLite "too many SQL variables" on large stores
**File:** `rag_cli.py` → `_restore_ingested_state()` (Chroma internals)\
**What:** On large Chroma stores the `_restore_ingested_state()` call fails with `OperationalError: too many SQL variables`. This prevents the pipeline from knowing which documents are already ingested, causing potential re-ingestion.\
**Fix:** Chunk the ID list into batches of 999 and call the query in a loop, or switch to a separate SQLite side-table that records ingested doc IDs outside of Chroma's internal queries.

---

## Output Quality — Large Testcases

### Q5 — Adaptive Tier 2 similarity threshold for large diverse species sets
**File:** `kg/graph_synthesizer.py` → `_query_tier2()`\
**What:** TC6 (11 species, taxonomically diverse) found only 4 Tier 2 embedding clusters vs. 29–30 for TC1-4. The fixed threshold of 0.78 is too strict for high-diversity sets where the same biological concept is expressed with different vocabulary across distantly related species. Synthesis had almost no convergence signal and returned 0 communities.\
**Proposed rule:** If `n_species > 6`, lower threshold from 0.78 → 0.72. Could also be made dynamic: start at 0.78, count clusters, if fewer than 10 found, retry at 0.72.\
**Risk:** Lower threshold means more false-positive similarity edges (spurious DOC_SIMILAR_TO). Needs evaluation on TC1-4 to confirm no regression.

---

## Ideas Worth Investigating

### I1 — Multi-agent system with state machine organizer
**What:** Replace the current linear pipeline (ingest → index → enrich → synthesize) with a multi-agent architecture where a state machine acts as the central orchestrator. Individual agents would be responsible for discrete tasks — e.g. a retrieval agent, an entity extraction agent, a hypothesis agent, a critique agent — and the state machine decides which agent to invoke next based on intermediate results and confidence signals.\

**Why it could help:** The current pipeline is stateless between stages — if synthesis produces a weak or generic result, nothing feeds back to improve retrieval or re-focus entity extraction. A state machine organizer could implement loops: if the synthesis agent returns low-confidence communities, the orchestrator could trigger a targeted re-retrieval agent for specific species, or invoke a critique agent to identify which species contributed noise, then re-run synthesis with adjusted inputs.\

**Relevant to current problems:**\
- Generic output for large N: a critique agent could flag communities covering all species and trigger focused re-querying\
- Vocabulary fragmentation (TC6 hypoxia): a gap-detection agent could notice that the expected signal is split across two vocabularies and attempt a bridging query\
- Literature sparsity: a coverage agent could assess per-species chunk counts and trigger additional ingestion for sparse species before synthesis\

**Open questions:** How much of the state machine logic is pre-programmed vs. emergent from LLM decisions? What are the state transition conditions? How do you prevent infinite loops on hard cases? What is the right granularity of agents — one per pipeline stage, or finer?\

**Risk/cost:** High complexity. Multi-agent systems are difficult to debug and the added orchestration overhead may not justify the benefit for simple testcases that already work well. Best evaluated on cases where the single-pass pipeline demonstrably fails (TC6 hypoxia, large-N testcases).\
