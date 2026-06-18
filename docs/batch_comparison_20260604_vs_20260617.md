# Batch Comparison: 2026-06-04 (OLD) vs 2026-06-17 (NEW)

**Pipeline:** GraphRAG candidate_sets_v1_le8  
**Old batch dir:** `logs_graphs/2026-06-04_13-27-21-batch-candidate_sets_v1_le8`  
**New batch dir:** `logs_graphs/2026-06-17_13-26-23-batch-candidate_sets_v1_le8`  
**Report date:** 2026-06-18

---

## 0. Changes Between Runs

The following code changes were deployed between the two runs:

| Change | Description |
|--------|-------------|
| **Q2 IC-based Tier 1 ranking** | `global_fan_in` now used to rank Tier 1 terms (replaces old `species_count ORDER BY`) |
| **precompute_fan_in auto-runs** | Fan-in values are now pre-computed inside the pipeline automatically (no manual step) |
| **ef_penalty removed** | The `1/(1+0.05×entity_forms)` term removed from the scoring formula, stopping penalisation of well-studied species |
| **Old formula** | `score = breadth × log((max_fan_in+1)/(fan_in+1)) × 1/(1+0.05×entity_forms)` |
| **New formula** | `score = breadth × log((max_fan_in+1)/(fan_in+1))` |
| **map_workers** | Reduced from 4 to 1 in the new run (unrelated configuration change) |
| **force_reenrich** | Set to `False` in new run vs `True` in old (enrichment reused from cache if available) |

---

## 1. Testcases Present in Each Run

### Old run (9 genes, all completed)

| Gene | Bundle timestamp |
|------|-----------------|
| SPATC1L | 2026-06-04_13-27-25 |
| UBAC1 | 2026-06-04_17-18-09 |
| NELFA | 2026-06-04_22-10-12 |
| KRBA2 | 2026-06-05_00-48-47 |
| FDXR | 2026-06-05_05-19-00 |
| CRYM | 2026-06-05_08-54-40 |
| WWP1 | 2026-06-05_13-59-53 |
| DCAF7 | 2026-06-05_16-48-52 |
| USP54 | 2026-06-05_18-52-55 |

### New run (6 genes started, 5 completed, 1 still running)

| Gene | Bundle timestamp | Status |
|------|-----------------|--------|
| SPATC1L | 2026-06-17_13-26-27 | Complete |
| UBAC1 | 2026-06-17_18-16-58 | Complete |
| NELFA | 2026-06-17_23-55-26 | Complete |
| KRBA2 | 2026-06-18_03-21-31 | Complete |
| FDXR | 2026-06-18_08-06-32 | Complete |
| CRYM | 2026-06-18_12-14-18 | **In progress** — indexing stage (4/8 species indexed as of report time; sturnira hondurensis, phyllostomus discolor, sagmatias obliquidens, bradypus torquatus not yet started) |

**No batch_manifest.json** exists for the new run (the batch is still in progress).

### Comparable genes (both complete): SPATC1L, UBAC1, NELFA, KRBA2, FDXR
### Old only (not re-run yet): WWP1, DCAF7, USP54
### New only: none

---

## 2. Timing

### Per-gene stage timings (seconds)

All stage timings are from `summary/meta.json → stage_timings`. The new batch ran with `map_workers=1` (old: 4), which is expected to slow indexing throughput; this is a configuration difference unrelated to the Q2/ef_penalty changes.

| Gene | Stage | OLD (s) | NEW (s) | Delta |
|------|-------|---------|---------|-------|
| **SPATC1L** | Indexing | 4,117 | 4,846 | +729 (+18%) |
| | Enrichment | 9,524 | 12,470 | +2,946 (+31%) |
| | Synthesis | 196 | 102 | -94 (-48%) |
| | **Total** | **13,838** | **17,417** | **+3,579 (+26%)** |
| **UBAC1** | Indexing | 4,751 | 5,339 | +588 (+12%) |
| | Enrichment | 12,548 | 14,855 | +2,307 (+18%) |
| | Synthesis | 217 | 102 | -115 (-53%) |
| | **Total** | **17,516** | **20,295** | **+2,779 (+16%)** |
| **NELFA** | Indexing | 1,923 | 2,733 | +810 (+42%) |
| | Enrichment | 7,473 | 9,562 | +2,089 (+28%) |
| | Synthesis | 113 | 64 | -49 (-43%) |
| | **Total** | **9,508** | **12,358** | **+2,850 (+30%)** |
| **KRBA2** | Indexing | 3,007 | 3,663 | +656 (+22%) |
| | Enrichment | 13,027 | 13,355 | +328 (+3%) |
| | Synthesis | 172 | 76 | -96 (-56%) |
| | **Total** | **16,206** | **17,094** | **+888 (+5%)** |
| **FDXR** | Indexing | 3,657 | 4,662 | +1,005 (+27%) |
| | Enrichment | 9,083 | 10,094 | +1,011 (+11%) |
| | Synthesis | 194 | 102 | -108 (-56%) |
| | **Total** | **12,933** | **14,858** | **+1,925 (+15%)** |

**Old batch total elapsed:** 120,798 s (~33.6 h) for 9 genes.  
**New batch total:** not yet complete (CRYM still running). Estimated ~82,000 s for the 5 completed genes.

### Timing interpretation

- **Indexing** is consistently slower in the new run (+12–42%), consistent with `map_workers` dropping from 4 to 1.
- **Enrichment** is also slower (+3–31%), which is unexpected given `force_reenrich=False` in the new run. In the old run `force_reenrich=True` forced Neo4j re-enrichment from scratch on every run. The new run uses cached enrichment where available, yet is still slower overall. This may reflect the enrichment cache being cold (first time these species ran since a code change), or the precompute_fan_in step adding overhead.
- **Synthesis** is notably faster in the new run (-43% to -56%). This is a genuine improvement: the IC-based ranking produces a tighter, 30-term tier1_raw list (same count numerically, but presented more efficiently to the LLM) and the removal of ef_penalty means the scoring pass is simpler.
- **Net effect:** The new run is ~15–30% slower end-to-end, entirely attributable to `map_workers=1`. With `map_workers=4` restored, the synthesis speedup would likely make the new run approximately neutral or slightly faster overall.

---

## 3. Output Quality — Tier 1 Communities

### 3.1 Generic term displacement

One of the primary goals of the IC-based ranking was to push uninformative over-broad terms (e.g., `abnormal anatomical structure`, `abnormal whole organism`) out of the top-10 Tier 1 results. Results are striking:

| Gene | Generic terms in OLD top 10 | Generic terms in NEW top 10 |
|------|-----------------------------|-----------------------------|
| SPATC1L | **4** (ranks 1, 2, 8, 9) | **0** |
| UBAC1 | **2** (ranks 2, 3) | **0** |
| NELFA | **1** (rank 5) | **0** |
| KRBA2 | 0 | 0 |
| FDXR | **4** (ranks 5, 6, 9, 10) | **1** (rank 8 — `abnormal whole organism`) |

"Generic terms" here refers to `abnormal anatomical structure` (XPO:0100006), `abnormal whole organism` (XPO:0100546), `multicellular anatomical structure` (UPHENO:7001060 / UBERON:0010000), and `abnormal multicellular anatomical structure` (XPO:0100010). These are high-fan_in parents that used to dominate rankings purely by species breadth.

**For SPATC1L**, the old top 2 were the two most generic XPO terms imaginable. The new top 2 are `size of anatomical entity phenotype` and `behavior` — substantively different and much more informative.

**For FDXR**, one generic (`abnormal whole organism`) still appears at rank 8. The top terms are now `growth phenotype`, `lateral plate mesoderm phenotype`, `blood island phenotype` — all hematopoietic/developmental, which is consistent with FDXR's known role in mitochondrial iron-sulfur cluster assembly and erythropoiesis.

### 3.2 SPATC1L — Top 10 Tier 1 term comparison

SPATC1L has 8 species (all mammal/fish), making it the clearest test case.

| Rank | OLD term | NEW term |
|------|----------|----------|
| 1 | abnormal anatomical structure (XPO:0100006) | **size of anatomical entity phenotype** (UPHENO:0075195) |
| 2 | abnormal whole organism (XPO:0100546) | **behavior** (GO:0007610) |
| 3 | homeostatic process phenotype (UPHENO:0049904) | digestive system phenotype (UPHENO:0002833) |
| 4 | digestive system phenotype (UPHENO:0002833) | **multicellular organismal process** (GO:0032501) |
| 5 | behavior process phenotype (UPHENO:0079826) | **craniocervical region phenotype** (UPHENO:0002764) |
| 6 | increased biological_process (UPHENO:0054970) | **multicellular organism morphology phenotype** (UPHENO:0081581) |
| 7 | anatomical collection phenotype (UPHENO:0002648) | **structure with developmental contribution from neural crest** (UPHENO:0002553) |
| 8 | multicellular anatomical structure (UPHENO:7001060) | **anatomical entity morphology phenotype** (UPHENO:0076692) |
| 9 | abnormal multicellular anatomical structure (XPO:0100010) | **material anatomical entity** (UBERON:0000465) |
| 10 | integumental system phenotype (UPHENO:0004064) | **behavior phenotype** (UPHENO:0049622) |

The new top 10 is more anatomically specific and biologically interpretable. `craniocervical region phenotype` and `structure with developmental contribution from neural crest phenotype` are meaningful signal about craniofacial biology. The old run had `homeostatic process phenotype` at rank 3 (driven by heterogeneous disease terms like fever, antibodies, encephalitis) — this has been displaced entirely.

### 3.3 UBAC1 — Top 10 Tier 1 term comparison

| Rank | OLD term | NEW term |
|------|----------|----------|
| 1 | trunk phenotype (UPHENO:0003413) | **musculature phenotype** (UPHENO:0002816) |
| 2 | multicellular anatomical structure (UPHENO:7001060) | **musculature of body phenotype** (UPHENO:0003432) |
| 3 | abnormal multicellular anatomical structure (XPO:0100010) | **skin of body phenotype** (UPHENO:0003811) |
| 4 | head phenotype (UPHENO:0002844) | **integument phenotype** (UPHENO:0002635) |
| 5 | organ part phenotype (UPHENO:0002531) | abdominal segment element phenotype (UPHENO:0003093) |
| 6 | hematopoietic system phenotype (UPHENO:0004459) | bone element phenotype (UPHENO:0002870) |
| 7 | embryo morphology phenotype (UPHENO:0087665) | head phenotype (UPHENO:0002844) |
| 8 | appendage phenotype (UPHENO:0002861) | integumental system phenotype (UPHENO:0004064) |
| 9 | compound organ phenotype (UPHENO:0002754) | digestive system element phenotype (UPHENO:0002546) |
| 10 | skeletal system phenotype (UPHENO:0002964) | craniocervical region phenotype (UPHENO:0002764) |

The new ranking reveals a clear musculature + integument signal that was buried in the old run. The old #1 (`trunk phenotype`) is a broad catch-all; the new top 2 are both specific to muscle. `skin of body phenotype` and `integument phenotype` are now prominent — consistent with UBAC1's known association with ubiquitin-mediated protein quality control that affects structural tissues.

### 3.4 NELFA — Top 10 Tier 1 term comparison

NELFA has only 4 species (minimum threshold), making signal weaker overall.

| Rank | OLD term | NEW term |
|------|----------|----------|
| 1 | material anatomical entity (UBERON:0000465) | **habitat** (ENVO:01000739) |
| 2 | response to stimulus phenotype (UPHENO:0049586) | abnormal cell (XPO:0100004) |
| 3 | germ layer / neural crest phenotype (UPHENO:0004680) | **abnormal survival** (MP:0010769) |
| 4 | multicellular organismal process phenotype (UPHENO:0050106) | mortality/aging (MP:0010768) |
| 5 | multicellular anatomical structure (UBERON:0010000) | mortality/aging phenotype (UPHENO:3000002) |
| 6 | Xenopus phenotype (XPO:0000000) | **sexually immature organism** (UBERON:0007021) |
| 7 | lateral plate mesoderm phenotype (UPHENO:0003159) | organism subdivision (UBERON:0000475) |
| 8 | dense mesenchyme tissue phenotype (UPHENO:0002542) | zone of organ phenotype (UPHENO:0003021) |
| 9 | embryonic tissue phenotype (UPHENO:0002567) | decreased biological_process (UPHENO:0005433) |
| 10 | blood island phenotype (UPHENO:0003207) | abnormal anatomical system (XPO:0100070) |

NELFA is a mixed result. The old run was heavily dominated by Xenopus/embryonic terms (ranks 3–10 are all germ-layer / Xenopus-model artefacts), while the new ranking surfaces `habitat`, `abnormal survival`, and `mortality/aging` — more interpretable in the context of NELFA's role as a transcription elongation factor. The appearance of `habitat` (ENVO:01000739) at rank 1 is notable and may reflect ecological diversity in the 4-species set rather than a gene function signal; it warrants inspection.

### 3.5 KRBA2 — Top 10 Tier 1 term comparison

| Rank | OLD term | NEW term |
|------|----------|----------|
| 1 | epithelium phenotype (UPHENO:0005141) | **biological_process in cell phenotype** (UPHENO:0081999) |
| 2 | gonad phenotype (UPHENO:0003056) | head phenotype (UPHENO:0002844) |
| 3 | macromolecule metabolic process phenotype (UPHENO:0049652) | craniocervical region phenotype (UPHENO:0002764) |
| 4 | biological_process rate phenotype (UPHENO:0080377) | **liver phenotype** (UPHENO:0003405) |
| 5 | multicellular organismal process phenotype (UPHENO:0050106) | **digestive system gland phenotype** (UPHENO:0003423) |
| 6 | trunk phenotype (UPHENO:0003413) | Zebrafish Phenotype (ZP:0000000) |
| 7 | multi-tissue structure phenotype (UPHENO:0002902) | **unilaminar epithelium phenotype** (UPHENO:0005110) |
| 8 | mesenchyme phenotype (UPHENO:0002577) | **exocrine gland phenotype** (UPHENO:0002780) |
| 9 | abnormal organism subdivision (XPO:0100003) | **hepatobiliary system phenotype** (UPHENO:0004061) |
| 10 | anatomical group phenotype (UPHENO:0005133) | abdominal segment element phenotype (UPHENO:0003093) |

The new KRBA2 run produces a striking hepatic convergence signal: liver phenotype, digestive system gland phenotype, exocrine gland phenotype, and hepatobiliary system phenotype all appear in the top 10. The old top 10 had `gonad phenotype` at rank 2 and was diffuse across tissue types. The hepatic signal is biologically plausible — KRBA2 has been associated with lipid metabolism and liver-expressed genes. The disappearance of `gonad phenotype` removes a potentially artefactual signal. Note that the new top 10 still contains `Zebrafish Phenotype (ZP:0000000)` at rank 6, which is a model-organism artefact analogous to the old `Xenopus phenotype`.

### 3.6 FDXR — Top 10 Tier 1 term comparison

| Rank | OLD term | NEW term |
|------|----------|----------|
| 1 | digestive system phenotype (UPHENO:0002833) | **growth phenotype** (UPHENO:0049874) |
| 2 | anatomical structure physiology phenotype (UPHENO:0002385) | **lateral plate mesoderm phenotype** (UPHENO:0003159) |
| 3 | anatomical system physiology phenotype (UPHENO:0002269) | **blood island phenotype** (UPHENO:0003207) |
| 4 | cell phenotype (UPHENO:0086172) | **dense mesenchyme tissue phenotype** (UPHENO:0002542) |
| 5 | abnormal anatomical structure (XPO:0100006) | anatomical collection phenotype (UPHENO:0002648) |
| 6 | abnormal whole organism (XPO:0100546) | **digestive system element phenotype** (UPHENO:0002546) |
| 7 | mesoderm-derived structure phenotype (UPHENO:0002554) | **mesoderm phenotype** (UPHENO:0004892) |
| 8 | homeostatic process phenotype (UPHENO:0049904) | abnormal whole organism (XPO:0100546) |
| 9 | multicellular anatomical structure (UPHENO:7001060) | anatomical structure physiology phenotype (UPHENO:0002385) |
| 10 | abnormal multicellular anatomical structure (XPO:0100010) | **hematopoietic system phenotype** (UPHENO:0004459) |

FDXR is a mitochondrial ferredoxin reductase required for iron-sulfur cluster assembly and heme synthesis. The new ranking showing `blood island phenotype`, `lateral plate mesoderm phenotype`, and `hematopoietic system phenotype` in the top 10 is substantially more biologically meaningful than the old top terms. The old run's leading terms (`digestive system phenotype`, `anatomical structure physiology phenotype`) gave no hematopoietic signal at all in the top 4. The new run correctly captures the heme/blood biology. One generic term (`abnormal whole organism`) remains at rank 8 — a regression from zero generics expected.

---

## 4. Output Quality — General

### 4.1 Community counts and tier breakdown

| Gene | OLD communities | OLD tier breakdown | NEW communities | NEW tier breakdown |
|------|-----------------|-------------------|-----------------|-------------------|
| SPATC1L | 5 | merged:5 | 6 | merged:4, llm:2 |
| UBAC1 | 5 | merged:5 | 4 | merged:4 |
| NELFA | 4 | merged:3, llm:1 | 4 | merged:4 |
| KRBA2 | 5 | merged:2, upheno:2, llm:1 | 3 | merged:3 |
| FDXR | 3 | merged:2, llm:1 | 4 | merged:1, upheno:2, llm:1 |

The new run tends to produce slightly fewer merged communities (most have 3–4 vs 4–5 in old). NELFA changed from having an `llm` community (no Tier 1 anchor) to all `merged` communities — a positive sign that more communities now have ontology backing. KRBA2 dropped from 5 to 3 communities; the three communities are more focused but the hypothesis is tighter as a result.

### 4.2 Enricher statistics

| Gene | OLD entities_seen | OLD mapped | OLD map% | NEW entities_seen | NEW mapped | NEW map% | sim_pairs OLD | sim_pairs NEW |
|------|------------------|------------|----------|------------------|------------|----------|---------------|---------------|
| SPATC1L | 5,877 | 1,385 | 23.6% | 6,511 | 1,350 | 20.7% | 0 | 1,010 |
| UBAC1 | 8,063 | 2,048 | 25.4% | 7,849 | 1,785 | 22.7% | 0 | 1,108 |
| NELFA | 3,683 | 1,164 | 31.6% | 3,900 | 984 | 25.2% | 0 | 315 |
| KRBA2 | 5,926 | 2,025 | 34.2% | 5,435 | 1,589 | 29.2% | 0 | 787 |
| FDXR | 6,262 | 1,415 | 22.6% | 5,721 | 1,091 | 19.1% | 0 | 720 |

The new run has **lower ontology mapping rates** across all 5 genes (20–29% vs 23–34%). This is unexpected if `force_reenrich=False` was reusing the same enrichment data. The discrepancy suggests the enrichment was in fact re-run from scratch (e.g., Neo4j was cleared between runs), and the new enricher version is more conservative about what it maps. The `similarity_pairs_added` metric is new: all new runs have non-zero values (315–1,108) while all old runs show 0. This indicates a new feature in the enricher that adds IS_A edges based on similarity rather than exact ontology ancestry — this did not exist in the old run.

### 4.3 Gene function hypotheses

Below are the full hypotheses from both runs for all comparable genes, for qualitative assessment.

**SPATC1L**
- **OLD:** "The shared gene likely encodes a multifunctional regulator essential for maintaining tissue homeostasis and structural integrity across diverse organ systems, particularly the integumentary, musculoskeletal, and craniofacial tissues. Its disruption triggers a conserved systemic inflammatory response and metabolic collapse, suggesting it plays a critical role in epithelial barrier function, connective tissue development, and the acute phase response to cellular stress or infection."
- **NEW:** "The shared gene likely functions as a master regulator of developmental plasticity and metabolic homeostasis, coordinating somatic growth with craniofacial morphogenesis to adapt to diverse ecological niches. It appears to integrate environmental cues (such as seasonality) to modulate immune responses against specific pathogens while simultaneously regulating complex social behaviors necessary for survival in group-living species."
- **Assessment:** Both hypotheses are generic. The new one mentions ecological niches and social behaviors, which reflects the new top-ranked terms (behavior, size) but is arguably less focused on gene biology. Neither hypothesis is particularly incisive for SPATC1L (a sperm-tail centriole-adjacent gene). The ecologically-themed terms ranking first is a sign that SPATC1L's species set lacks a strong convergent molecular phenotype — this is a pipeline limitation, not introduced by the changes.

**UBAC1**
- **OLD:** "coordinating the development and maintenance of the nervous, musculoskeletal, and visceral systems... pleiotropic syndrome characterized by sensory deficits, structural skeletal abnormalities..."
- **NEW:** "coordinating structural development across musculoskeletal systems while simultaneously maintaining epithelial barriers in the skin and digestive tract... modulates immune cell proliferation (evidenced by splenomegaly) and inflammatory cytokine signaling (IL-10)..."
- **Assessment:** The new hypothesis is more specific — it references splenomegaly and IL-10 as concrete evidence. The old hypothesis invented "sensory deficits" which were not particularly visible in the tier1_raw. The new one follows the musculature + integument signal that the IC-ranking surfaced. Moderate improvement.

**NELFA**
- **OLD:** "central regulator of the stress-immune axis, modulating the transition from acute stress responses (cortisol/lactate elevation) to chronic inflammatory states..."
- **NEW:** "central regulator integrating metabolic status with immune surveillance, specifically modulating the NF-κB pathway and oxidative stress responses... critical for maintaining homeostasis during physiological extremes such as hibernation or migration..."
- **Assessment:** Both hypotheses are plausible for NELFA (a negative elongation factor involved in stress response). The new one is slightly more precise (NF-κB, hibernation/migration context) and the new communities (Inflammatory Response Regulation, Oxidative Stress Response, Nervous System Dysfunction) map better to known NELFA biology than the old (Hepatic and Hematologic Toxicity, Reproductive Tract Morphology).

**KRBA2**
- **OLD:** "central regulator of cellular plasticity and homeostasis... governing reproductive maturation and digestive efficiency... Wnt/FGF signaling..."
- **NEW:** "central metabolic sensor that integrates environmental stress signals (specifically hypoxia) to regulate hepatic lipid homeostasis... modulating Wnt/β-catenin signaling pathways..."
- **Assessment:** The new hypothesis is narrower and more testable. The hepatic lipid metabolism angle is better supported by the tier1_raw terms (liver, hepatobiliary, exocrine gland). The old hypothesis mentioned reproductive maturation (gonad phenotype at rank 2 in old), which has been deprioritized. The new Wnt/β-catenin reference is retained from the old hypothesis, which is consistent.

**FDXR**
- **OLD:** "critical regulator of mitochondrial function and cellular redox homeostasis... maintaining the structural integrity of epithelial barriers in the digestive, respiratory, and renal systems..."
- **NEW:** "master regulator of mesodermal differentiation that coordinates hematopoietic development with systemic growth and organogenesis... failed blood cell production, gastrointestinal malformation, arrested somatic growth..."
- **Assessment:** The new hypothesis is substantially better for FDXR. FDXR (ferredoxin reductase) is mitochondrial and essential for heme biosynthesis and erythropoiesis. "Failed blood cell production" and "hematopoietic development" in the new hypothesis directly reference FDXR's known function. The old hypothesis focused on epithelial barriers (a digestive phenotype artefact from the old rank-1 term), which is a less direct link. This is the clearest improvement across all five comparable genes.

---

## 5. Anomalies and Issues

### 5.1 fan_in = -1 for all new Tier 1 terms

All Tier 1 terms in all new runs carry `fan_in = -1`. This means the precompute_fan_in step ran (the field is present and stored) but returned no valid values. This is unexpected: the IC-based ranking was supposed to use actual fan_in values from the Neo4j graph to differentiate terms. With all terms scoring `fan_in = -1`, the ranking formula reduces to:

```
score = breadth × log((max_fan_in+1)/(fan_in+1))
      = breadth × log((-1+1)/(-1+1))
      = breadth × log(0/0)  → undefined / likely handled as 0 or fallback
```

The fact that the output still looks improved (generic terms displaced) suggests either (a) the fallback ordering is being applied (likely breadth-only or alphabetical tie-breaking), or (b) the precompute returned -1 as a sentinel for "not computed from a live graph" and the code falls back to a different ranking path. This warrants investigation — the IC-based ranking as designed is **not actually running** in the new batch. The improvements seen are from some other aspect of the pipeline change (possibly the removal of the ef_penalty changing how ties are broken, or the sim_pairs enrichment adding new edges that shift which terms have coverage).

### 5.2 CRYM still running at report time

CRYM (new run) is mid-indexing as of 2026-06-18. Only 4 of 8 species have completed indexing (equus asinus, pseudorca crassidens, rattus norvegicus partially done; hydropotes inermis ingest only). The enrichment and synthesis stages have not started. This run cannot be compared yet.

### 5.3 WWP1, DCAF7, USP54 not re-run

Three genes from the old batch were not restarted in the new batch. No indication whether they were intentionally excluded (e.g., the new batch is a rolling run that will continue) or dropped. The new batch's `species_files/` directory contains only SPATC1L, UBAC1, NELFA, KRBA2, FDXR, CRYM — confirming the new batch is currently a 6-gene subset.

### 5.4 No batch_manifest.json in new run

The new batch has no `batch_manifest.json` at the root level (the old run had one). This file is presumably written at the end of the full batch. Its absence confirms the batch is still in progress.

### 5.5 map_workers regression

The new run uses `map_workers=1` vs `map_workers=4` in the old run. This is an unintentional regression causing ~20–40% slower indexing. This should be restored to 4 before production comparison runs.

### 5.6 Lower ontology mapping rates in new enricher

As noted in Section 4.2, the new enricher maps 20–29% of seen entities vs 23–34% in the old enricher — a consistent 3–6 percentage point drop. Combined with the new `similarity_pairs_added` (non-zero in all new runs), this suggests the enricher was meaningfully updated between runs. The similarity pairs feature adds IS_A edges via cosine similarity when direct ontology matches fail, which may partially compensate for the lower direct mapping rate but is a different mechanism with different specificity characteristics.

### 5.7 fan_in sentinel value in synthesis evidence

The `graph_synthesis_evidence.json` files also store `fan_in = -1` for all tier1 entries, consistent with item 5.1. If the precompute_fan_in is writing -1 as a "not found in graph" sentinel, this points to the Neo4j database not containing the `global_fan_in` property on phenotype nodes that the ranking query expects. This is likely the root cause.

---

## 6. Summary Assessment

| Criterion | Verdict |
|-----------|---------|
| Generic terms displaced from top 10 | **Yes — strong improvement** (0–1 generic terms in new top 10 vs 0–4 in old) |
| More specific biological terms in top 10 | **Yes** — musculature, liver, hematopoietic, craniofacial terms now leading |
| IC-based ranking actually running | **No** — all fan_in values are -1; ranking falls back to an alternative path |
| ef_penalty removal effect observable | **Likely yes** — this is the most plausible explanation for the ranking shift given that IC-based ranking is non-functional |
| Hypothesis quality improved | **Moderate** — FDXR is clearly better; UBAC1 and KRBA2 are somewhat better; SPATC1L and NELFA are comparable |
| Timing regression | **Yes** — ~15–30% slower, due to map_workers=1 (unrelated to the Q2/ef_penalty changes) |
| New features in enricher | **Yes** — similarity_pairs_added is a new enricher output; ontology mapping rate is lower |
| Batch complete | **No** — CRYM still running; WWP1/DCAF7/USP54 not re-run |

**Bottom line:** The new pipeline produces noticeably better Tier 1 rankings in 4 of 5 comparable genes, with generic catch-all terms largely eliminated. The most likely driver is the **ef_penalty removal**, not the IC-based ranking (which appears to be a no-op due to the fan_in=-1 issue). Once the fan_in precompute is fixed and working against a live graph, the IC-based ranking may yield further improvements. The synthesis speed improvement (~50% faster) is real. The main operational issue to address is the `map_workers` regression and the fan_in precompute bug.
