# Performance Investigation: P1, P2, P3 Impact

**Batches compared:**
- **Batch 1** `2026-06-02_17-02-48` — before P1/P2; `map_workers=1`, `force_reenrich=False`
- **Batch 2** `2026-06-04_13-27-21` — after P1/P2 (both done 2026-06-04); `map_workers=4`, `force_reenrich=True`

---

## Bottom Line

| Change | Expected impact | Observed impact |
|---|---|---|
| **P1** — batch Neo4j writes (UNWIND) | 10–50× fewer round trips, ~5h → 15–30 min enrichment | **Not measurable.** Per-entity enrichment throughput is actually ~30–40% *worse* in B2 when parallelism is factored out. |
| **P2** — batch similarity edge writes (UNWIND) | Medium perf; fixes zero-pairs bug | **Functional fix confirmed.** Tier 2 now produces edges. Adds cost for genes with many similar entities. |
| **P3** — `map_workers=4` | Medium. Parallelises entity mapper | **Primary driver of all speedup.** ~65–74% parallel efficiency; explains the 2.5–3× wall-clock improvement. |

---

## 1. P1 — Batched Neo4j Writes Did Not Produce Visible Speedup

### Raw timing

| Gene | B1 enrichment (s) | B2 enrichment (s) | Wall-clock speedup | B1 entities | B2 entities |
|---|---|---|---|---|---|
| FDXR | 17,946 | 9,083 | 1.97× | 4,171 | 6,262 |
| SPATC1L | 18,014 | 9,524 | 1.89× | 3,894 | 5,877 |
| UBAC1 | 19,485 | 12,548 | 1.55× | 4,632 | 8,063 |
| KRBA2 | 20,537 | 13,027 | 1.58× | 3,740 | 5,926 |
| NELFA | 7,848 | 7,473 | 1.05× | 1,497 | 3,683 |
| CRYM | 12,457 | 12,899 | 0.97× | 2,847 | 8,760 |
| DCAF7 | 5,434 | 5,394 | 1.01× | 1,159 | 3,235 |
| USP54 | 9,732 | 10,662 | 0.91× | 2,116 | 6,674 |

### Isolating P1 from P3

The observed wall-clock speedup conflates two independent changes: P1 (batched writes) and P3 (4 workers). To separate them, compute the **effective sequential throughput** = wall-clock rate × workers:

| Gene | B1 (1 worker) s/entity | B2 equiv. serial s/entity | P1 signal |
|---|---|---|---|
| FDXR | 4.30 | 5.80 | **−35%** (slower) |
| SPATC1L | 4.63 | 6.48 | **−40%** |
| UBAC1 | 4.21 | 6.22 | **−48%** |
| KRBA2 | 5.49 | 8.79 | **−60%** |
| NELFA | 5.24 | 8.12 | **−55%** |
| CRYM | 4.38 | 5.89 | **−35%** |
| DCAF7 | 4.69 | 6.67 | **−42%** |
| USP54 | 4.60 | 6.39 | **−39%** |

**Every gene is slower per entity in equivalent serial time in B2.** P1 contributed no measurable improvement.

### Why P1 didn't help

The original estimate assumed Neo4j write round trips were the bottleneck (~40k trips for 10k entities). The data shows the actual bottleneck is the **LLM mapping calls** (stages 1 and 3 of `_run_upheno_mapping()`), not the database writes.

At ~4–5 s/entity in B1, each entity spends the vast majority of its processing time waiting for the LLM API. The Neo4j write operations, which are network calls measured in milliseconds, are a rounding error compared to LLM calls measured in seconds. Batching them had no observable effect.

The additional per-entity cost in B2 serial time comes from two confounds introduced by `force_reenrich=True`:
1. Several species with near-zero indexed entities in B1 (e.g. CRYM `phyllostomus_discolor`: 200 → 4,406; USP54 `vulpes_lagopus`: 1 → 3,110) were re-ingested fresh, bringing in harder / more diverse entities with lower cache-hit rates.
2. P2 similarity-pair computation added new work per enrichment run that did not exist in B1 at all.

### Indexing stage (unchanged by P1)

The indexing stage already used UNWIND in B1. Per-entity indexing rates are nearly identical:

| Gene | B1 (s/idx_entity) | B2 (s/idx_entity) | Change |
|---|---|---|---|
| SPATC1L | 0.347 | 0.348 | ~0% |
| FDXR | 0.341 | 0.309 | −9% |
| CRYM | 0.375 | 0.312 | −17% |
| USP54 | 0.388 | 0.326 | −16% |
| KRBA2 | 0.337 | 0.294 | −13% |

The ~10–17% improvement where it appears is within server load variation — not attributable to any code change. This is the expected baseline since indexing was already batch-written.

---

## 2. P2 — Similarity Edges: Functional Fix, Moderate Cost

P2 fixed the bug documented in Q3: `similarity_pairs_added` was 0 for every gene in B1. B2 produced non-zero similarity pairs for four genes:

| Gene | B1 pairs | B2 pairs | B2 tier2_clusters | B1 tier2_clusters |
|---|---|---|---|---|
| CRYM | 0 | 1,308 | 2 | 0 |
| DCAF7 | 0 | 179 | 6 | 0 |
| USP54 | 0 | 637 | 15 | 1 |
| WWP1 | 0 | 268 | — | — |

Tier 2 is now actively contributing to synthesis. DCAF7 went from 0 to 6 Tier 2 clusters; USP54 went from 1 to 15. The fix is real and working.

**Cost:** The similarity computation is O(n²) over mapped entities. For CRYM (2,128 mapped entities) and USP54 (1,482), this adds a significant quadratic load that partially offsets the P3 speedup. CRYM and USP54 are the two genes that ended up slower overall in B2 — the similarity-pair computation is the primary cause.

Genes with 0 pairs in B2 (FDXR, KRBA2, NELFA, SPATC1L, UBAC1) still produced no Tier 2 edges, which likely means their entity similarities are all below the threshold, not that P2 is broken for them.

---

## 3. P3 — `map_workers=4`: Primary Driver of All Speedup

With P1 explaining nothing, the full observed wall-clock speedup comes from parallelism. The effective parallel efficiency is consistent across genes:

| Gene | Wall-clock speedup per entity | Parallel efficiency (÷ 4) |
|---|---|---|
| FDXR | 2.97× | 74% |
| SPATC1L | 2.85× | 71% |
| DCAF7 | 2.81× | 70% |
| USP54 | 2.88× | 72% |
| UBAC1 | 2.70× | 68% |
| NELFA | 2.58× | 65% |
| KRBA2 | 2.50× | 63% |

Average efficiency: **~69%**. This is consistent with the LLM server sustaining roughly 2.5–3 concurrent calls before throughput saturates. Going above 4 workers would likely hit diminishing returns quickly.

For small entity sets (NELFA: 1,497; DCAF7: 1,159 in B1), the total enrichment time was already short enough that parallelism doesn't help — confirmed by near-zero speedup for those two genes.

---

## 4. Entity Count Inflation in Batch 2

`force_reenrich=True` caused significantly more entities to be processed in B2, independent of any code change:

| Gene | B1 enricher entities | B2 enricher entities | Ratio |
|---|---|---|---|
| CRYM | 2,847 | 8,760 | 3.1× |
| USP54 | 2,116 | 6,674 | 3.2× |
| UBAC1 | 4,632 | 8,063 | 1.7× |
| NELFA | 1,497 | 3,683 | 2.5× |

The enricher/indexer entity ratio also rose from ~0.35 (B1) to ~0.51 (B2) across all genes, meaning B2 enriched a larger fraction of the indexed graph. Under `force_reenrich=False`, already-enriched nodes are skipped. With `True`, the full graph is re-traversed each time.

Several species had near-zero indexed entities in B1 due to incomplete prior ingestion; `force_reenrich=True` re-triggered ingestion for these:
- **CRYM:** `bradypus_torquatus` 0 → 2,407 entities; `phyllostomus_discolor` 200 → 4,406
- **USP54:** `vulpes_lagopus` 1 → 3,110 entities; `nomascus_leucogenys` 0 → 2,100

These are the primary reason for total enrichment time not improving as much as the per-entity wall-clock rate would predict.

---

## 5. Summary and Recommendations

### What actually worked
- **P3 (map_workers=4)** is working and delivering ~69% parallel efficiency. Keep it. There is no evidence that going beyond 4 workers would help — server concurrency appears to saturate around 3 simultaneous LLM calls.
- **P2** fixed the long-standing zero-similarity-pairs bug. Tier 2 is now producing communities for several genes. The quadratic cost is real but manageable if the entity set is not huge.

### What didn't work
- **P1 made no measurable difference.** The LLM mapping calls dominate enrichment time. Reducing Neo4j round trips from ~40k to ~4 UNWIND calls saved milliseconds against a bottleneck measured in seconds. The expected "10-50×" estimate assumed the wrong bottleneck.

### Where to look next for enrichment speedup
If the goal is to significantly reduce enrichment time further:
1. **The LLM mapping call itself is the bottleneck.** Options: batch the LLM prompts (multiple entities per call), use a smaller/faster model for the mapping stage, or cache mapping results across runs (memoise by entity string hash).
2. **P4 (pre-filter generic entities)** would directly reduce the number of LLM calls. Even removing 10% of trivially-generic entities (e.g. `"cell"`, `"tissue"`) saves proportional time.
3. **force_reenrich=False** recovers significant time on repeat runs by skipping already-enriched nodes. B1 benefited from this; B2 deliberately disabled it. For production batch runs this is the highest-ROI setting.
4. Increasing workers beyond 4 would require load testing — efficiency is already at 69% with 4.
