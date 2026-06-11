# Batch Run Comparison: 2026-06-02 vs 2026-06-04

**Batch 1:** `logs_graphs/2026-06-02_17-02-48-batch-candidate_sets_v1_le8`  
**Batch 2:** `logs_graphs/2026-06-04_13-27-21-batch-candidate_sets_v1_le8`  
**Genes analysed:** SPATC1L, UBAC1, NELFA, KRBA2, FDXR, CRYM, DCAF7, USP54, WWP1

---

## 1. Configuration Differences

The two batches differ on three settings that together explain most of the behavioural differences:

| Parameter | Batch 1 | Batch 2 |
|---|---|---|
| `map_workers` | 1 | 4 |
| `force_reenrich` | False | True |
| `similarity_pairs_added` | 0 (all genes) | non-zero for CRYM, DCAF7, USP54, WWP1 |

`force_reenrich=True` in Batch 2 means enrichment was re-run from scratch regardless of any cache. Combined with `map_workers=4`, this is effectively a parallelised re-enrichment run. The appearance of `similarity_pairs_added > 0` in Batch 2 for several genes but never in Batch 1 suggests a new enrichment feature was activated between the two runs.

---

## 2. Timing

### Per-gene total wall time (seconds)

| Gene | B1 total (s) | B2 total (s) | Δ (s) | B2 speedup | B1 enrichment | B2 enrichment |
|---|---|---|---|---|---|---|
| SPATC1L | 21,964 | 13,838 | −8,126 | **1.59×** | 18,015 | 9,524 |
| UBAC1 | 23,809 | 17,516 | −6,293 | **1.36×** | 19,485 | 12,548 |
| FDXR | 22,120 | 12,933 | −9,187 | **1.71×** | 17,946 | 9,083 |
| KRBA2 | 23,741 | 16,206 | −7,535 | **1.46×** | 20,537 | 13,027 |
| NELFA | 9,287 | 9,508 | +221 | ~1× | 7,848 | 7,473 |
| DCAF7 | 6,951 | 7,436 | +485 | ~1× | 5,434 | 5,394 |
| CRYM | 15,808 | 18,307 | **+2,498** | **0.86×** | 12,457 | 12,899 |
| USP54 | 12,254 | 14,862 | **+2,608** | **0.82×** | 9,732 | 10,662 |
| WWP1 | — (failed) | 10,132 | — | — | — | 7,304 |

### Interpretation

- The 4-worker enrichment is **clearly beneficial for large entity sets** (SPATC1L, UBAC1, FDXR, KRBA2 all show ~1.4–1.7× speedup driven almost entirely by shorter enrichment times).
- For **small entity sets** (NELFA, DCAF7) parallelism provides no measurable benefit — the work is too small to amortise the overhead.
- **CRYM and USP54 are slower in B2** despite 4 workers. Both also gained `similarity_pairs_added` in B2 (1,308 and 637 respectively), which represents additional work with no B1 counterpart. The indexing stage for CRYM also increased from 3,160 s to 5,305 s — worth investigating separately (see §5).

---

## 3. Results: Communities and Hypotheses

### Community count per gene

| Gene | B1 communities | B2 communities |
|---|---|---|
| SPATC1L | 4 | 5 |
| UBAC1 | 4 | 5 |
| NELFA | 4 | 4 |
| KRBA2 | 5 | 5 |
| FDXR | 4 | 3 |
| CRYM | 5 | 5 |
| DCAF7 | 4 | 3 |
| USP54 | 4 | 4 |
| WWP1 | 0 (failed) | 4 |

**Zero community labels are shared between B1 and B2 for any gene** — the LLM renamed all clusters across runs. However, thematic content is largely preserved for most genes (see below).

### Thematic consistency per gene

| Gene | B1 dominant themes | B2 dominant themes | Consistent? |
|---|---|---|---|
| SPATC1L | Neurological, Integumentary, Inflammation, Musculoskeletal | Inflammation, Craniofacial, Integumentary, Musculoskeletal, Digestive | ✅ Mostly |
| UBAC1 | Neurological, Inflammation, Skeletal, Metabolic | Neurological, Musculoskeletal, Digestive, Immune, Reproductive | ✅ Mostly |
| FDXR | Respiratory/Circulatory, Mitochondrial, GI, Neurological | Digestive/Respiratory, Mitochondrial, **Renal/Hematopoietic** | ⚠️ New renal theme in B2 |
| KRBA2 | Gonadal, Metabolic, Intestinal, Cardiovascular, Regenerative | Reproductive, Digestive, Stress/Metabolic, Cardiovascular, Regenerative | ✅ Very consistent |
| NELFA | Integumentary, Immune, Embryonic, Oxidative Stress | **Stress-Immune Dysregulation, Hepatic/Hematologic Toxicity, Reproductive Tract, Skin** | ⚠️ Diverged notably |
| CRYM | Digestive/Hepatic, Integumentary, Sensory/Craniofacial, Inflammation, Reproductive | Inflammation, Digestive, **Renal, Sensory, Neurological/Behavioral** | ⚠️ New renal + neuro in B2 |
| DCAF7 | Immune/Spleen, Gonadal, Intestinal, Epithelial | Immune/Spleen, Gonadal, Digestive | ✅ Consistent (B2 lost one community) |
| USP54 | Musculoskeletal, Immune, Neurological/Sensory, Renal/Vascular | **Integumentary Pigmentation, Renal, GI Microbiome, Sensory** | ❌ Large divergence |

**USP54 is the most divergent result.** B1 emphasises musculoskeletal and immune phenotypes; B2 foregrounds integumentary pigmentation and GI microbiome. This is likely a direct consequence of the enrichment re-run (`force_reenrich=True`) pulling in different entity sets (B2: 6,674 entities vs B1: 2,116).

### Gene function hypotheses

Both batches produce hypotheses in the same abstract register ("master regulator of…", "central coordinator of…"). The content tracks the community themes — hypotheses for KRBA2, FDXR, and DCAF7 are semantically similar between runs; NELFA, CRYM, and USP54 diverge more. No hypothesis directly contradicts its counterpart, but the level of specificity varies.

---

## 4. Input Differences

**Species input is identical for all 8 genes present in both batches.** The same canonical species and aliases were used. There are no input-level explanations for result differences — divergence comes entirely from the enrichment re-run and the similarity-pair feature.

---

## 5. Anomalies — Manual Review Recommended

### 5.1 WWP1 failed to complete in Batch 1
The `2026-06-04_01-29-01-wwp1-cb87bee9` directory contains only `species_input.json`. There is no `meta.json`, no `graph_synthesis.json`, no run outputs — the pipeline crashed or was killed before the enrichment stage started. The directory name timestamp (`2026-06-04_01-29-01`) falls inside the B1 batch window and immediately precedes DCAF7 (`01-31-29`), which did complete. **Check server logs around 2026-06-04 01:29 for an OOM or crash signal.**

### 5.2 CRYM and USP54 are slower despite 4 workers
CRYM: indexing grew from 3,160 s → 5,305 s (+68%). USP54: total time grew by 2,608 s. Both also gained `similarity_pairs_added` (1,308 and 637). These are the two largest similarity-pair counts in the entire batch. The similarity-pair computation is the likely bottleneck — it scales quadratically with the number of mapped entities. CRYM has the largest entity count in B2 (8,760 entities seen) and USP54 jumped from 2,116 to 6,674. **This feature should be profiled or made configurable if it causes unpredictable slowdowns.**

### 5.3 Much larger entity counts in Batch 2
B2 consistently ingests ~1.5–3× more entities than B1 across all genes (e.g. CRYM: 2,847 → 8,760; USP54: 2,116 → 6,674; NELFA: 1,497 → 3,683). Species inputs are identical, so this is driven by `force_reenrich=True` triggering a deeper or differently-seeded enrichment pass. It is not obvious from metadata alone whether this reflects better recall or additional noise. **The USP54 community divergence (§3) is the clearest case where the larger entity set produced substantially different top-level results — worth a manual biological plausibility check.**

### 5.4 New `similarity_pairs_added` feature in Batch 2
Batch 1 has `similarity_pairs_added = 0` for every gene. Batch 2 has non-zero values for CRYM (1,308), DCAF7 (179), USP54 (637), and WWP1 (268). This feature did not exist or was not triggered in B1. The genes where it fired (CRYM, USP54) are also the ones with the largest result divergence, which may be causal. **Verify whether this feature was intentionally enabled and whether it is producing valid similarity edges.**

### 5.5 FDXR gained a renal/hematopoietic community in B2
B1 FDXR: respiratory, mitochondrial, GI, neurological. B2 FDXR: digestive/respiratory, mitochondrial, **renal and hematopoietic** (tier `llm`). The `llm` tier label means this community was inferred by the LLM without a direct HPO/UPHENO anchor. FDXR is known to be involved in mitochondrial electron transfer, so a renal phenotype is plausible but less canonical. **Worth a spot-check of the supporting entities in `graph_synthesis_evidence.json`.**

---

## Summary

| Topic | Key Finding |
|---|---|
| Speed | B2 is faster for large jobs (4-worker enrichment gives 1.4–1.7× on FDXR, KRBA2, SPATC1L, UBAC1); negligible or negative for small jobs |
| Results | Thematically consistent for most genes; **USP54 and NELFA diverge substantially** |
| Input | No input differences — species sets are identical in both batches |
| Anomalies | WWP1 crash in B1; CRYM/USP54 slowdown from similarity-pair computation; large entity-count inflation in B2; new similarity-pair feature needs validation |
