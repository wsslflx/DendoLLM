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
**Poor — expected signal missed.** The hypoxia signal is partially present but buried: `hypoxia` appears as a supporting entity inside "Integumentary and Thermal Adaptation", and "Stress Response and Metabolism" (Tier 3 only) mentions hypoxia, mitochondria, and glucocorticoid — which is essentially the correct signal but unlabeled and without ontology support.

Compare with testcase1 where hypoxia formed a dedicated, well-evidenced community. The difference: in testcase1 all 4 species had dense, focused underground-oxygen literature. Here the signal is split between diving physiology (whales, dugong — "breath-hold diving", "myoglobin") and high-altitude physiology (phrynocephalus, triplophysa — "altitude hypoxia") which use different vocabulary and don't cluster together at the entity level. Additionally the high-altitude species are data-sparse, so the high-altitude half of the hypothesis barely registers.

---

## Summary table

| Testcase | Species | Expected signal | Signal found | Tier | Quality |
|---|---|---|---|---|---|
| 1 — Subterranean mammals | 4 | Visual reduction | Partial (3/4 species) | LLM only | Partial |
| 3 — Color-change | 4 | Pigmentation/chromatophores | Yes | Tier 1+2 | Good |
| 4 — Electroreception | 7 | Electrosensory organs | Yes (5/7 species) | Tier 1 | Moderate |
| 5 — Mixed mammals | 12 | Unknown | N/A — all generic | — | Poor |
| 6 — Hypoxia adaptation | 11 | Low-oxygen tolerance | Buried/diluted | LLM only | Poor |

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
