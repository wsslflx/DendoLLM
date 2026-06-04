# Graph Pipeline Run Analysis

Analysis of five testcase runs using the GraphRAG pipeline (graph indexing with `granite4.1:8b`, synthesis with `qwen3.5:27b`).

---

## Cross-run technical notes

- **Entity extraction**: `granite4.1:8b` was the actual model used for all runs (default after fix), despite `meta.json` incorrectly logging `qwen2.5:7b` — a known logging bug in `run_graph_pipeline.py`.
- **`similarity_pairs_added: 0`** across all runs: DOC_SIMILAR_TO edges were never created, so Tier 2 evidence is absent in most communities. Root cause not yet investigated.
- **No stdout/stderr logs are saved**: if entity extraction silently fails mid-run there is no way to diagnose it post-hoc. A log file per species would be useful.
- **Community tiers**: `merged` = Tier 1 + LLM; `llm` = Tier 3 only (no ontology backing).

---

## Testcase 1 — Subterranean mammals
**Species:** blind mole-rat, cape golden mole, star-nosed mole, naked mole-rat
**Expected signal:** blindness / reduced visual system

### Entity extraction
| Species | Chunks indexed | Entities | Triples |
|---|---|---|---|
| Nannospalax ehrenbergi | 656 | 2776 | 1248 |
| Chrysochloris asiatica | 633 | 2319 | 910 |
| Condylura cristata | 391 | 1318 | 662 |
| Heterocephalus glaber | 299 | 1344 | 732 |

Healthy extraction rates (~3–4 entities/chunk). Junk filter caught 0–11 chunks per species.

### Enrichment
4630 entities seen, 1706 mapped to uPheno (37%). Reasonable for biological text.

### Communities
| Label | Tier | Species | Tier 1 | Tier 2 |
|---|---|---|---|---|
| Craniofacial and Dental Adaptation | merged | 4/4 | cranial skeletal system phenotype | — |
| Hypoxia and Metabolic Tolerance | merged | 4/4 | response to hypoxia phenotype | yes |
| **Visual System Reduction** | **llm** | **3/4** | **—** | **—** |
| Neural Crest and Brain Plasticity | merged | 4/4 | central nervous system phenotype | — |
| Dermatological and Tissue Integrity | merged | 4/4 | tissue phenotype | yes |

### Assessment
**Partially good.** The expected signal (visual reduction) appeared but is weak: Tier 3 only, covers only 3/4 species, no ontology backing. The uPheno ontology simply does not cover visual reduction phenotypes well enough to produce Tier 1 evidence. The other communities (hypoxia, skin, craniofacial) are biologically real and well-supported — underground mammals genuinely share these pressures — but are non-specific to the gene of interest.

---

## Testcase 3 — Color-change species
**Species:** octopus, panther chameleon, gray tree frog, western rock lobster
**Expected signal:** color change / chromatophores / camouflage

### Entity extraction
| Species | Chunks indexed | Entities | Triples |
|---|---|---|---|
| Octopus vulgaris | 711 | 2274 | 1035 |
| Furcifer pardalis | 355 | 2003 | 789 |
| Dryophytes versicolor | 510 | 1534 | 668 |
| Panulirus cygnus | 225 | 794 | 354 |

Best entity extraction rates across all runs. Highest uPheno mapping rate (40%).

### Enrichment
11696 entities seen, 4648 mapped (40%).

### Communities
| Label | Tier | Species | Tier 1 | Tier 2 |
|---|---|---|---|---|
| Reproductive System Morphology | merged | 4/4 | testis phenotype | yes |
| **Integumentary and Pigmentation** | **merged** | **4/4** | **abnormal epidermal pigmentation** | **yes** |
| Neurological and Behavioral Control | merged | 4/4 | central nervous system phenotype | yes |
| Metabolic and Growth Regulation | merged | 4/4 | whole organism weight, abnormal | yes |
| Vascular and Circulatory Integrity | llm | 4/4 | — | — |

### Assessment
**Good.** The pigmentation/chromatophore signal came through clearly with both Tier 1 and Tier 2 evidence — the strongest result across all testcases for signal specificity. The neural/behavioral community is also plausible since color change is neurally driven.

The top-ranked community (Reproductive System Morphology) is likely **spurious**: four phylogenetically distant animals (invertebrate, reptile, amphibian, crustacean) appearing to share gonadal phenotypes suggests the entity extraction is picking up generic biology text rather than species-specific traits. This should be treated as noise.

---

## Testcase 4 — Electroreception species
**Species:** electric eel, elephantnose fish, great white shark, electric ray, channel catfish, axolotl, platypus
**Expected signal:** electroreception / electrosensory organs

### Entity extraction
| Species | Chunks indexed | Entities | Triples |
|---|---|---|---|
| Ambystoma mexicanum | 870 | 3195 | 1630 |
| Ornithorhynchus anatinus | 944 | 3203 | 1436 |
| Carcharodon carcharias | 613 | 2427 | 1071 |
| Ictalurus punctatus | 335 | 1156 | 406 |
| Gnathonemus petersii | 342 | 1013 | 421 |
| Torpedo marmorata | 188 | 801 | 333 |
| Electrophorus electricus | 326 | 550 | 225 |

Electrophorus electricus has a low entity yield (~1.7/chunk vs ~3.5 average) — likely sparse literature coverage for this species specifically.

### Enrichment
6093 entities seen, 1716 mapped (28%).

### Communities
| Label | Tier | Species | Tier 1 | Tier 2 |
|---|---|---|---|---|
| Regenerative Morphogenesis | merged | 5/7 | — | — |
| **Electrocyte and Neural Development** | **merged** | **5/7** | **nervous system phenotype** | **—** |
| Wound Healing and Stress Response | merged | 7/7 | response to stimulus phenotype | — |
| Musculoskeletal Morphology | merged | 7/7 | — | — |

### Assessment
**Moderate.** The electroreception signal appears explicitly in "Electrocyte and Neural Development" (electrocytes, electric organ discharge, electroreceptors), which is a correct hit. However it covers only 5/7 species — great white shark and channel catfish did not contribute strongly enough.

The dominant problem is **axolotl bias**: axolotl has the most entities (3195) and its regeneration literature is so rich that it creates an entirely off-target community ("Regenerative Morphogenesis") that outranks the actual signal. Including axolotl in an electroreception testcase is a design question — it has only weak electroreceptive ability and its literature will always pull towards regeneration. "Wound Healing" and "Musculoskeletal" communities are generic noise driven by axolotl and platypus literature volume.

---

## Testcase 5 — Mixed mammals (12 species)
**Species:** bats (2), seals (2), big cats (2), sheep, vampire bat, mole, grey whale, lion, arctic fox
**Expected signal:** unknown

### Entity extraction — notable issues
| Species | Chunks | Entities | Entities/chunk |
|---|---|---|---|
| Ovis aries | 1998 | 8425 | 4.2 |
| Artibeus jamaicensis | 1374 | 4530 | 3.3 |
| Eschrichtius robustus | 1066 | 3699 | 3.5 |
| Lynx canadensis | 1098 | 828 | **0.75** |
| Vulpes lagopus | 1786 | 3182 | 1.8 (583 junk chunks, 33%) |

Lynx canadensis has the worst entity extraction ratio of any species across all runs — the model struggled with its literature. Vulpes lagopus had 583/1786 chunks (33%) flagged as junk, suggesting the ingested content contained many tables/reference lists.

### Communities
All 5 communities cover all 12 species — a clear sign of generic output:

| Label | Tier | Species | Tier 1 | Tier 2 |
|---|---|---|---|---|
| Craniofacial and Dental Morphology | merged | 12/12 | face phenotype | — |
| Somatic Growth and Body Size | merged | 12/12 | growth phenotype | — |
| Limb and Appendage Development | merged | 12/12 | limb bud phenotype | — |
| Central Nervous System and Sensory Organs | merged | 12/12 | CNS phenotype | — |
| Reproductive and Gonadal Development | llm | 12/12 | — | — |

### Assessment
**Poor specificity.** When every community covers all 12 species, the signal is generic mammalian biology, not a shared gene-specific trait. Craniofacial, growth, limbs, CNS — these would appear for any arbitrary set of 12 mammals. The `min_species` threshold of 6 (N//2) filtered out anything more specific. No Tier 2 evidence in any community.

The data quality issues (lynx, arctic fox) further weaken the run. This testcase illustrates the core scaling problem: with 12 species the pipeline reliably produces generic output.

---

## Testcase 6 — Hypoxia-adapted species (11 species)
**Species:** phrynocephalus forsythii, triplophysa siluroides, right whale, fin whale, dugong, sperm whale, lophiomys imhausi, phyllotis (genus), musk deer, beaver, hippo
**Expected signal:** adaptation to low-oxygen environments (diving or high-altitude species)

### Entity extraction — notable issues
| Species | Chunks | Entities |
|---|---|---|
| Dugong dugon | 1107 | 3431 |
| Moschus moschiferus | 1073 | 3395 |
| Physeter macrocephalus | 708 | 1646 |
| Phrynocephalus forsythii | 96 | 517 |
| Triplophysa siluroides | 83 | 391 |

The two high-altitude species (phrynocephalus, triplophysa) are data-sparse with only 83–98 chunks each — very little literature available. They contribute minimally to the shared signal.

**Additional data quality concerns:**
- `phyllotis` is listed as genus only (no species name) — may have caused mixed/incorrect data ingestion
- `hippopotamus amphibius kiboko` is a subspecies — the extra epithet may have confused source queries

### Enrichment
8660 entities seen, 1846 mapped (21%) — lowest mapping rate across all runs.

### Communities
| Label | Tier | Species | Tier 1 | Tier 2 |
|---|---|---|---|---|
| Integumentary and Thermal Adaptation | merged | 11/11 | integument phenotype | — |
| Craniofacial and Neural Morphology | merged | 11/11 | — | — |
| Visceral Organ Pathology | merged | 11/11 | — | — |
| Stress Response and Metabolism | llm | 10/11 | — | — |

### Assessment
**Poor — expected signal buried, not surfaced.** The hypoxia signal is present in the data but diluted across two communities:

- **"Integumentary and Thermal Adaptation"** (Tier 1, 11/11 species): `hypoxia` appears as a *supporting entity* buried inside a community dominated by skin/coat phenotypes. The community label and Tier 1 anchor are about integument, not oxygen tolerance — the hypoxia entity was swept into this community because the diving mammals' skin literature co-occurs with metabolic adaptation text.
- **"Stress Response and Metabolism"** (Tier 3 / LLM only, 10/11 species): explicitly mentions hypoxia, mitochondria, and glucocorticoid — essentially the correct signal — but has no ontology backing (Tier 1/2 absent) and was ranked last. The LLM correctly identified the shared pattern but the graph evidence was too weak to produce a named community.

Why the signal got buried: the testcase mixes two vocabularies for the same underlying biology. Diving species (whales, dugong) use terms like "myoglobin", "breath-hold diving", "diving reflex"; high-altitude species (phrynocephalus, triplophysa) use "altitude hypoxia", "erythrocyte count", "HIF pathway". These don't cluster in Tier 1 because their surface forms differ — and Tier 2 (embedding similarity) is broken (see recurring problems). Additionally the high-altitude species have only 83–98 chunks each, so their vocabulary is too sparse to anchor a community. The result is the correct signal split into fragments that individually fall below the `min_species` threshold for a standalone community.

---

## Testcase 7 — SPATC1L (production run)
**Species:** stoat, spotted gar, Ovis ammon polii (×2 including hybrid), topi, striped hyena, monk saki, European badger
**Expected signal:** inner ear hair cell development / stereocilia organization / hearing
**Run time:** 6.1 h total (indexing 63 min, enrichment 300 min, synthesis 3 min)

### Entity extraction
| Species | Docs | Chunks | Entities | Triples |
|---|---|---|---|---|
| Damaliscus lunatus | 7 | 678 | 3176 | 1111 |
| Lepisosteus oculatus | 7 | 405 | 1948 | 724 |
| Ovis ammon polii | 10 | 388 | 1386 | 545 |
| Mustela erminea | 5 | 334 | 1320 | 474 |
| Hyaena hyaena | 7 | 377 | 1044 | 372 |
| Meles meles | 6 | 209 | 757 | 408 |
| Pithecia pithecia | 5 | 216 | 652 | 233 |
| Ovis ammon polii × Ovis aries | 4 | 124 | 579 | 257 |

Document counts are very low (4–10 per species). The hybrid subspecies *Ovis ammon polii × Ovis aries* is particularly sparse at 4 docs.

### Enrichment
3894 entities seen, 820 mapped to uPheno (21%). Similarity pairs added: 0 (Tier 2 still broken).

### Communities
| Label | Tier | Species | Tier 1 | Tier 2 |
|---|---|---|---|---|
| Neurological and Craniofacial Defects | merged | 8/8 | abnormal anatomical structure | — |
| Integumentary and Pigmentation Disorders | merged | 8/8 | integument phenotype | — |
| Systemic Inflammation and Immune Response | merged | 8/8 | abnormal blood/immune | — |
| Musculoskeletal and Locomotor Defects | merged | 8/8 | musculoskeletal | — |

### Assessment
**Poor — expected signal absent.** All 4 communities cover all 8 species, a clear sign of generic output. SPATC1L is associated with inner ear hair cell development and stereocilia; no community related to hearing, ciliary structure, or mechanosensation appeared.

The most likely cause is insufficient literature: 4–10 documents per species is too few to produce species-specific signal. With so little text, the entities extracted are dominated by generic biology (anatomy, inflammation, skeletal defects) that appears in any vertebrate's Wikipedia/PMC profile. A SPATC1L-relevant article would need to be among the handful of documents retrieved for each species — an unlikely coincidence at this document count.

The run also confirms the enrichment time problem: 5 hours for 8 species with only ~2,700 chunks total is far too slow for the 66-testcase batch.

---

## Summary table

| Testcase | Species | Expected signal | Signal found | Tier | Quality |
|---|---|---|---|---|---|
| 1 — Subterranean mammals | 4 | Visual reduction | Partial (3/4 species) | LLM only | Partial |
| 3 — Color-change | 4 | Pigmentation/chromatophores | Yes | Tier 1+2 | Good |
| 4 — Electroreception | 7 | Electrosensory organs | Yes (5/7 species) | Tier 1 | Moderate |
| 5 — Mixed mammals | 12 | Unknown | N/A — all generic | — | Poor |
| 6 — Hypoxia adaptation | 11 | Low-oxygen tolerance | Buried in wrong community | LLM only | Poor |
| 7 — SPATC1L (production) | 8 | Stereocilia / hearing | Not found | — | Poor |

---

## Recurring problems

### 1. Too many species → generic output
`min_species = max(2, N//2)` means Tier 1 requires a trait to appear in half the species. For 11–12 species this threshold is so high that only universal mammalian biology survives. The synthesis LLM also averages across N subgraph summaries and gravitates to the broadest shared patterns. See `min_species` discussion in `graph_synthesizer.py`.

### 2. Tier 2 never fires
`similarity_pairs_added: 0` in all runs. DOC_SIMILAR_TO edges are not being created during enrichment. This removes an entire evidence tier from all results.

### 3. Species with imbalanced literature dominate
When one species has 3–4× more entities than others (axolotl in TC4, ovis aries in TC5), its literature themes dominate the shared communities even if those themes are irrelevant to the gene of interest.

### 4. Data-sparse species contribute little
Species with <100 chunks (phrynocephalus, triplophysa in TC6) barely register in synthesis. Their signal — which may be the key shared trait — is drowned out by better-documented species.

### 5. Vocabulary fragmentation
The same biological concept expressed in different vocabulary across species (diving: "myoglobin", "breath-hold"; altitude: "hypoxia tolerance", "erythrocyte count") does not cluster in Tier 1 because the surface forms differ. Tier 2 (embedding similarity) is designed to bridge this — but it is not working (see point 2).

---

## Current production problems and discussed solutions

### Problem A — Output quality: too many species produces generic traits

**Root cause:** With N species, `min_species = N//2` requires a trait to appear in half of them. For N=8+ this filter is so strict that only pan-mammalian biology (craniofacial anatomy, growth, CNS) survives. The synthesis LLM further dilutes signal by averaging across all species' subgraph summaries. SPATC1L (N=8) and TC5 (N=12) both produced all-species communities with no gene-specific signal.

**Observed threshold:** Results are useful up to ~N=7 (TC4 electroreception, moderate quality). At N=8 (SPATC1L) and above, output becomes generic. The production dataset has 66 testcases with ≤8 unique species, so many are near or at this limit.

**Discussed solutions:**

1. **Species sub-clustering (preferred):** For testcases with N>7, split species into overlapping sub-groups of 3–5, run synthesis on each sub-group independently, then merge/deduplicate the resulting communities. This keeps `min_species` at a level where specific signal can survive while still covering all species.

2. **Lower `min_species` threshold:** Pass `--min-species 2` to require only 2 species to agree. This recovers more specific signals but increases noise — communities may reflect one species' idiosyncratic literature rather than a shared trait.

3. **Fix Tier 2 (DOC_SIMILAR_TO edges):** Tier 2 is designed to bridge vocabulary fragmentation across species. If it worked, the same concept expressed differently (e.g. "stereocilia" vs "hair bundle") would still cluster. All runs so far show `similarity_pairs_added: 0` — root cause not yet investigated.

---

### Problem B — Speed: enrichment is the dominant bottleneck

**Root cause:** The SPATC1L run took 6.1h total, of which **5h was enrichment** for just 3,894 entities across 8 species. The bottleneck is `_push_mapping()` in `graph_enricher.py`: it opens **3 separate Neo4j sessions per entity** (MERGE OntologyTerm node, MERGE DOC_MAPPED_TO edge, SET upheno_enriched flag), plus **2 more sessions per IS_A ancestor**. At ~10k entities with ~3 ancestors each, that is ~40,000 individual Neo4j round trips run serially. The similarity edge step (`_run_similarity_edges`) similarly writes one session per passing pair inside an O(n²) loop.

**Already implemented optimizations:**
- SQLite persistent cache in `trait_mapper.py` — cache hits skip all 3 LLM stages; reused across all testcases
- Cosine early-exit: auto-accept ≥0.95, auto-reject <0.50 — skips Stage 3 LLM for obvious cases
- `--map-workers` for parallel trait mapping (ThreadPoolExecutor)
- `--norm-batch-size` for batched Stage 1 normalization (N traits per LLM call)
- `--index-workers` for parallel chunk entity extraction
- UNWIND batch writes for triple creation in indexer
- Batch already-indexed check (single Neo4j query)
- Surface form deduplication (same text mapped once)

**Discussed solutions (not yet implemented):**

1. **Batch Neo4j writes in enricher using UNWIND (highest impact):** Collect all (entity → ontology term) mappings first, then push in 2 UNWIND queries (one for all OntologyTerm nodes, one for all DOC_MAPPED_TO edges + upheno_enriched flags). Same pattern already used in the indexer. Estimated impact: 10–50× fewer round trips, likely cuts enrichment from 5h to ~15–30 min.

2. **Batch similarity edge writes using UNWIND:** Collect all pairs above threshold into a list, write as a single UNWIND query instead of one session per pair.

3. **Increase `--map-workers`:** Run was executed with default of 1 worker. Increasing to 4 would parallelize the LLM-heavy Stage 1/3 mapping calls. Stages 2 (cosine, CPU) and cache hits would benefit immediately regardless of LLM server capacity.

4. **Pre-filter generic entity surface forms:** Entities like `"cell"`, `"tissue"`, `"protein"` are too generic to produce useful uPheno mappings and will auto-reject or map to high-level useless ancestors. Skipping them reduces LLM call count and Neo4j noise.
