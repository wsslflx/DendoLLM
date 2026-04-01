# Pipeline Version Comparison

This document describes the differences between all pipeline versions in this project.
**v4 is the current production version.** All others are kept as reference in `pipeline/archive/`.

---

## Quick Reference

| Version | Orchestrator | Inventory | RAG Strategy | Unique Feature |
|---------|-------------|-----------|--------------|----------------|
| v1 | `run_full_pipeline.py` | `inventory_single_2.py` | Single query + MMR | Baseline multi-run |
| v2 | `run_full_pipeline_v2.py` | `inventory_single_2.py` | Single query + MMR | Mechanism inference phase |
| v2b | `run_full_pipeline_v2b.py` | (calls v2 internally) | Single query + MMR | Proposer–verifier dual track |
| v3 | `run_full_pipeline_v3.py` | `inventory_single_3.py` | Dual query + dedup | Dual-query RAG + internal synthesis |
| v4 | `run_full_pipeline_v4.py` | `inventory_single_4.py` | Single query + MMR + normalization | Semantic trait clustering |

---

## v1 — Baseline Multi-Run Pipeline

**Entry point:** `pipeline/archive/run_full_pipeline.py`
**Inventory:** `pipeline/inventory_single_2.py`
**Output root:** `logs_v1/`

### What it does
The original pipeline. Runs N independent inventory iterations over a species list, then aggregates results.

**Stages:**
1. For each run: call `inventory_single_2.py` per species (RAG extraction → trait JSON)
2. Aggregate source statistics across runs
3. `analyze_runs.py` — compute trait frequencies and citation drift
4. `group_traits_llm.py` — LLM synonym grouping → `trait_groups.json`

### Inventory: `inventory_single_2.py`
- Single semantic query per species
- MMR retrieval to diversify chunks
- fcntl file lock on ingestion to serialize Wikipedia/PMC fetching
- Outputs per-species trait JSON + ingestion logs (`ingested_docs.json`, `used_chunks.json`)

### When to use
Simplest and fastest. Good baseline and for quick iteration on prompts.

---

## v2 — Mechanism Inference

**Entry point:** `pipeline/archive/run_full_pipeline_v2.py`
**Inventory:** `pipeline/inventory_single_2.py` (same as v1)
**Output root:** `logs_v2/`

### What it does
Adds a **mechanism inference phase** after trait aggregation. After extracting traits across runs, an LLM proposes candidate biological mechanisms that could explain the observed trait pattern.

**Stages (after inventory, same as v1):**
1. Trait frequency analysis
2. **Inference phase** — LLM receives high-frequency traits and proposes mechanistic hypotheses, filtered by `--min-species-support` and `--min-source-support` thresholds
3. Grouping

### Key additions over v1
- `--min-species-support` — minimum number of species a trait must appear in
- `--min-source-support` — minimum citation support required
- `--min-inferred-traits`, `--max-inferred-traits` — bounds on inferred hypothesis count
- `--skip-ingest-all` — alternative to skip all ingestion

### When to use
When you want the pipeline to not just extract traits but also propose *why* those traits co-occur (gene function hypotheses).

---

## v2b — Proposer–Verifier Dual Track

**Entry point:** `pipeline/archive/run_full_pipeline_v2b.py`
**Inventory:** Calls `run_full_pipeline_v2.py` (Track A) as a subprocess
**Output root:** `logs_v2b/`

### What it does
A two-track architecture on top of v2:

- **Track A** (inherited): standard RAG extraction + mechanism inference (v2 pipeline)
- **Track B** (new): independent proposer–verifier loop
  - *Proposer LLM* generates candidate trait claims from retrieved documents
  - *Verifier LLM* scores each claim against evidence (separate model possible)
  - Claims deduplicated by token Jaccard similarity
  - Final synthesis combines Track A and Track B results

### Key additions over v2
- `--track-a-only` — run only the Track A extraction (skip verifier)
- `--proposer-model`, `--verifier-model` — can use different LLMs for each role
- `--claim-verification-threshold` — minimum verifier confidence to accept a claim
- `--min-proposal-support` — minimum runs a claim must appear in

### When to use
When you want a second independent evidence-checking pass on top of standard extraction, or want to experiment with separate proposer/verifier models. Most complex to configure.

---

## v3 — Dual-Query RAG + Internal Synthesis

**Entry point:** `pipeline/archive/run_full_pipeline_v3.py`
**Inventory:** `pipeline/archive/inventory_single_3.py`
**Output root:** `logs_v3/`

### What it does
Redesigns the RAG retrieval step to use **two semantic queries** per species instead of one, capturing a broader range of trait-relevant documents. Synthesis and inference are handled inside the orchestrator (not the inventory script).

**Stages:**
1. For each run: `inventory_single_3.py` — dual-query extraction per species
2. Synthesis phase (inside orchestrator) — LLM synthesis across species per run
3. Inference phase — mechanism hypotheses
4. Aggregation

### Inventory: `inventory_single_3.py`
- **Query 1:** `"{species} morphology behavior ecology sensory phenotype"`
- **Query 2:** `"{species} natural history trait field observations habitat use diet foraging locomotion coloration morphometric"`
- Deduplicates chunks across both queries (by content hash + doc_id + chunk index)
- Per-document chunk cap: max 10 chunks from the same source
- Runs its own internal synthesis step before returning

### Key additions over v1/v2
- Dual-query retrieval finds more diverse, relevant source passages
- Per-document cap prevents any single paper from dominating context
- Synthesis embedded in orchestrator gives tighter control over prompt

### When to use
When retrieval quality is the bottleneck — species with sparse or domain-specific literature benefit from broader query coverage.

---

## v4 — Semantic Trait Normalization (Current)

**Entry point:** `pipeline/run_full_pipeline_v4.py`
**Inventory:** `pipeline/inventory_single_4.py`
**Output root:** `logs_v4/`

### What it does
Adds a **semantic normalization layer** between raw extraction and synthesis. After traits are extracted (using the v1/v2 inventory), semantically similar traits are clustered and merged into canonical forms before synthesis. This reduces noise and duplicate traits across runs.

**Stages:**
1. For each run: `inventory_single_4.py`
   - Calls `inventory_single_2.py` (v1 extraction) → raw `open_traits[]`
   - `build_hybrid_species_profile()` — embeds all traits, clusters by cosine similarity
   - Merges traits within threshold into a `normalized_tags[]` with aggregated sources and support counts
   - Infers `latent_factors[]` — higher-order biological patterns across normalized traits
2. Cross-species synthesis on normalized profiles
3. `analyze_runs.py` + `group_traits_llm.py` (same as v1)

### Key additions over v1
- `--hybrid-sim-threshold` (default: `0.82`) — cosine similarity cutoff for merging traits
- Hybrid profile JSON per species contains:
  - `open_traits[]` — raw extracted traits
  - `normalized_tags[]` — merged canonical traits with `support_count` and `mean_confidence`
  - `latent_factors[]` — inferred higher-order biological factors

### When to use
**Recommended default.** The normalization step significantly reduces synonym noise across runs (e.g. "reduced eyesight" and "vestigial visual system" merge into one cluster), leading to cleaner synthesis output.

---

## Inventory Scripts

### `inventory_single.py` — Original (unused)
Minimal first version. Single query, no locking, no structured logging. Kept as historical reference only.

### `inventory_single_2.py` — Stable extraction base
The workhorse used by v1, v2, v2b, and v4. Single query + MMR, fcntl ingestion locking, structured source logging. Supports `--reuse-traits` and `--skip-ingest` flags.

### `inventory_single_3.py` — Dual-query variant
Used only by v3. Dual-query retrieval with deduplication and per-document chunk cap. Also runs an internal synthesis step.

### `inventory_single_4.py` — Normalization wrapper
Used by v4. Wraps `inventory_single_2.py` and adds the hybrid normalization step. Not a standalone extractor — depends on v2 extraction underneath.

---

## Choosing a Version

```
Need quick results / prompt testing?      → v1
Want mechanism hypotheses?                → v2
Want claim verification?                  → v2b
Retrieval quality is the bottleneck?      → v3
Best overall quality (recommended)?       → v4
```
