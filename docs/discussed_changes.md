# Discussed but Not Yet Implemented Changes

Changes discussed during development that have not been implemented yet.
Mark as done with `**Done YYYY-MM-DD**` when implemented.

---

## Performance

### P1 — Batch Neo4j writes in enricher (UNWIND)
**Done 2026-06-04**
**File:** `kg/graph_enricher.py` → `_push_mapping()`
**Expected impact:** Highest. Currently opens 3 separate Neo4j sessions per entity (MERGE OntologyTerm, MERGE DOC_MAPPED_TO edge, SET upheno_enriched) plus 2 more per IS_A ancestor. At ~10k entities with ~3 ancestors each this is ~40k individual round trips run serially. Fix: collect all mappings first, push in 2 UNWIND queries (one for all OntologyTerm nodes, one for all edges + flags). Same pattern already used in `graph_indexer.py`. Estimated impact: 10–50× fewer round trips, likely cuts enrichment from ~5h to ~15–30 min for 8 species.
**Implemented:** Replaced per-entity `_push_mapping()` loop in `_run_upheno_mapping()` with 4 UNWIND queries (A: MERGE OntologyTerm nodes, B: MERGE DOC_MAPPED_TO + mark enriched, C: mark no-match, D: ancestor nodes + IS_A edges grouped by rel_type). Also deduplicates ancestors by term_id before owlready2 lookup.

### P2 — Batch similarity edge writes (UNWIND)
**Done 2026-06-04**
**File:** `kg/graph_enricher.py` → `_run_similarity_edges()`
**Expected impact:** Medium. Currently writes one `session.run()` per passing pair inside the O(n²) loop. Fix: collect all pairs above threshold into a list, write as a single UNWIND query.
**Implemented:** Collects all passing pairs into `pairs_to_write`, writes in chunks of 1000 via UNWIND.

### P3 — Increase `--map-workers` in batch runs
**File:** `scripts/run_batch_tsv.py`, `pipeline/run_graph_pipeline.py`
**Expected impact:** Medium. All batch runs so far used default of 1 worker. The trait mapper has Stage 2 (cosine, CPU-only) and cache-hit paths that don't touch the LLM at all and benefit immediately from parallelism. Even LLM-bound Stage 1/3 benefit if the server handles light concurrency. Needs testing to find the right value without overloading the server.

### P4 — Pre-filter generic entity surface forms before uPheno mapping
**File:** `kg/graph_enricher.py` → `_run_upheno_mapping()`
**Expected impact:** Low-medium. Entities like `"cell"`, `"tissue"`, `"protein"`, `"gene"` are too generic to map to a useful specific uPheno term. They consume LLM calls and produce high-level useless ancestors that add noise to Tier 1. A blocklist or length/frequency filter applied before `map_traits_batch()` would skip them. Also improves Tier 1 signal quality.

---

## Output Quality — KG Node Reranking

### Q1 — Ontology depth scoring for Tier 1 (early convergence preference)
**File:** `kg/graph_synthesizer.py` → `_query_tier1()`
**What:** The Tier 1 Cypher query traverses `IS_A*0..3` but treats all ancestor nodes equally regardless of depth. A node 1 hop from the leaf (specific term) ranks the same as a node 3 hops up (generic). Fix: return minimum traversal depth per ancestor from the Cypher query, use it as a ranking signal — prefer ancestors where species converge at a low depth (early in the hierarchy = more specific match). Implementation: add `min(depth)` to the WITH clause, include in ORDER BY as a secondary sort key after species count.
**Risk:** Low. Only affects which Tier 1 candidates are passed to the LLM and their order. No downstream breakage.

### Q2 — Global fan-in scoring for Tier 1 (ontology node specificity)
**File:** `kg/graph_synthesizer.py` → `_query_tier1()`, plus a one-time offline precomputation
**What:** The current `--tier1-max-entity-forms` (default 50) is a crude proxy for fan-in — it counts how many distinct entity surface forms in the current run map to an ancestor, and discards anything above the threshold. A better approach: precompute the global fan-in (total distinct leaf OntologyTerms pointing to each ancestor across the full uPheno ontology) once offline and store it on the OntologyTerm node. Then use it as a continuous ranking score rather than a hard cutoff. Nodes reachable from virtually any biological entity (e.g. `"abnormal anatomical structure"`) get a low score; nodes reachable from only 10–30 leaf terms get a high score.
**Why this matters:** The reproductive system community in TC3 (UPHENO:0002523, "testis phenotype") passed the current `max_entity_forms` filter because it only had ~8 entity forms in that run — but globally it is reachable from any animal reproductive anatomy mention. A global fan-in score would have down-ranked it.
**Prerequisite:** One-time offline computation from the uPheno OWL file. Cache result in the ontology SQLite or as a JSON file. No live pipeline impact once cached.
**Risk:** Medium. Aggressive penalization could suppress valid broad signals (e.g. TC1 visual reduction was already weak). Expose fan-in weight as a CLI parameter.

---

## Output Quality — Signal Specificity

### Q3 — Fix `similarity_pairs_added: 0` (Tier 2 DOC_SIMILAR_TO edges never created)
**File:** `kg/graph_enricher.py` → `_run_similarity_edges()`
**What:** Zero DOC_SIMILAR_TO edges have been created across all runs. Root cause not yet investigated. Tier 2 is designed to bridge vocabulary fragmentation — the same concept expressed differently across species (e.g. "myoglobin" vs "breath-hold diving") would cluster via embedding similarity. Without it, the hypoxia signal in TC6 split into fragments that individually fell below `min_species`. Fixing this is prerequisite for meaningful `"merged"` tier communities.
**Status:** Root cause unknown. Needs investigation before implementation.

### Q4 — Species sub-clustering for large testcases (N > 7)
**File:** New module or extension to `pipeline/run_graph_pipeline.py`
**What:** For testcases with N > 7 species, `min_species = N//2` becomes so strict that only pan-mammalian biology survives. Fix: split species into overlapping sub-groups of 3–5, run synthesis on each sub-group independently, then merge/deduplicate the resulting communities. Communities that appear in multiple sub-groups are promoted as high-confidence. This keeps `min_species` at a level where specific signal can survive while still covering all species.
**Risk:** Medium-high. Requires new orchestration logic and a community deduplication/merging step. Needs careful design to avoid double-counting species or inflating community scores.

---

## Ideas Worth Investigating

### I1 — Multi-agent system with state machine organizer
**What:** Replace the current linear pipeline (ingest → index → enrich → synthesize) with a multi-agent architecture where a state machine acts as the central orchestrator. Individual agents would be responsible for discrete tasks — e.g. a retrieval agent, an entity extraction agent, a hypothesis agent, a critique agent — and the state machine decides which agent to invoke next based on intermediate results and confidence signals.

**Why it could help:** The current pipeline is stateless between stages — if synthesis produces a weak or generic result, nothing feeds back to improve retrieval or re-focus entity extraction. A state machine organizer could implement loops: if the synthesis agent returns low-confidence communities, the orchestrator could trigger a targeted re-retrieval agent for specific species, or invoke a critique agent to identify which species contributed noise, then re-run synthesis with adjusted inputs.

**Relevant to current problems:**
- Generic output for large N: a critique agent could flag communities covering all species and trigger focused re-querying
- Vocabulary fragmentation (TC6 hypoxia): a gap-detection agent could notice that the expected signal is split across two vocabularies and attempt a bridging query
- Literature sparsity: a coverage agent could assess per-species chunk counts and trigger additional ingestion for sparse species before synthesis

**Open questions:** How much of the state machine logic is pre-programmed vs. emergent from LLM decisions? What are the state transition conditions? How do you prevent infinite loops on hard cases? What is the right granularity of agents — one per pipeline stage, or finer?

**Risk/cost:** High complexity. Multi-agent systems are difficult to debug and the added orchestration overhead may not justify the benefit for simple testcases that already work well. Best evaluated on cases where the single-pass pipeline demonstrably fails (TC6 hypoxia, large-N testcases).
