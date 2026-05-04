# GraphRAG Pipeline Overview

> What happens when you run:
> ```bash
> python pipeline/run_graph_pipeline.py --species-file testcase1.json --runs 1
> ```

---

```
run_graph_pipeline.py --species-file testcase1.json --runs 1
│
├─► Read species file
│     └── 4 species: blind mole-rat, cape golden mole,
│                    star-nosed mole, naked mole-rat
│
├─► FOR EACH species (run 1):
│     │
│     ├─► Document Ingestion
│     │     └── Fetch papers from Wikipedia / PMC / OpenAlex → Chroma vectorstore
│     │
│     ├─► Graph Indexing  (GraphRAG)
│     │     └── LLM reads each chunk → extracts entities & relations
│     │               → stores DocChunk / DocEntity / MENTIONS / RELATED_TO in Neo4j
│     │
│     ├─► Graph Retrieval  (GraphRAG)
│     │     └── Traverse Neo4j to collect all entities + triples for this species
│     │
│     ├─► Trait Extraction  (LLM)
│     │     └── LLM reads structured graph context → outputs trait list per species
│     │
│     └─► Hybrid Normalization
│           └── Embed traits, cluster by similarity → normalized tags + latent factors
│
├─► uPheno Enrichment  (across all species at once)
│     ├─► Map every extracted entity → uPheno / HP / MP ontology term
│     ├─► Add ontology ancestor hierarchy (IS_A edges) into Neo4j
│     └─► Compute cross-species entity similarity → DOC_SIMILAR_TO edges in Neo4j
│
├─► Three-Tier Cross-Species Synthesis
│     │
│     ├─► Tier 1 — Ontology Convergence  (graph query, no LLM)
│     │     └── Which entities from different species map to the same ontology ancestor?
│     │           → high-precision biological communities
│     │
│     ├─► Tier 2 — Semantic Similarity  (graph query, no LLM)
│     │     └── Which entities from different species are near-synonyms
│     │           (DOC_SIMILAR_TO edges)?
│     │           → catches synonyms Tier 1 misses
│     │
│     └─► Tier 3 — LLM Synthesis  (LLM)
│           ├─► Merge Tier 1 + Tier 2 communities that describe the same biology
│           ├─► Gap-fill: spot additional shared conditions from entity lists
│           ├─► Write 1-2 sentence biological interpretation per community
│           └─► Write gene function hypothesis integrating all communities
│                         → graph_synthesis.json
│
└─► Legacy KG Step  (unchanged from v4 pipeline)
      └── Species/Trait/OntologyTerm KG built from extracted traits
                → existing downstream analysis
```

---

## Key Conceptual Shift vs. Old Pipeline

| Old pipeline | New pipeline |
|---|---|
| LLM reads raw text chunks | LLM reads a pre-structured graph of entities & relations |
| Cross-species comparison = LLM reads all trait lists | Cross-species comparison = pre-computed via ontology + embeddings in Neo4j, LLM only interprets |
| Synonym problem handled by LLM alone | Synonym problem handled by embedding similarity (Tier 2) before LLM sees anything |

---

## Output Files (per bundle)

| File | Produced by |
|---|---|
| `traits/<species>.json` | Trait Extraction (per species) |
| `runs/run_01/<species>/graph_context.txt` | Graph Retrieval |
| `graph_enricher_summary.json` | uPheno Enrichment |
| `graph_synthesis.json` | Three-Tier Synthesis |
| `summary/meta.json` | Pipeline orchestrator |
