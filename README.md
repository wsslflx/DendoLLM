# DendoLLM

A biological species trait extraction pipeline using GraphRAG + LLMs. It ingests scientific
literature (Wikipedia, PMC, OpenAlex papers), builds a knowledge graph of entities and relations
per species in Neo4j, maps entities onto biomedical ontologies (uPheno/HP/MP), and synthesizes
traits shared across a set of species — all driven by an Ollama-backed LLM.

**Biological goal:** generate hypotheses about gene function. Given a set of species that share
a gene of interest, the pipeline extracts their common biological traits. If species sharing a
gene consistently show a trait (e.g. electroreception, subterranean lifestyle), that trait becomes
a hypothesis about what the gene does.

**Evaluating output quality:** there is no automated test suite — outputs are evaluated by eye at
the end of a run. Good output is biologically plausible, not overfitted to what the LLM already
"knows" about the gene, and backed by traceable sources. `testcases/` contains a few worked
examples with an expected signal, e.g. `testcase1.json` (subterranean mammals → expect a
blindness/reduced-visual-system signal) and `testcase4.json` (electroreception species → expect
electric/mechanosensory receptor signal).

## Pipeline overview

```
run_graph_pipeline.py --species-file testcases/testcase1.json --runs 1
│
├─► Read species file
│
├─► FOR EACH species:
│     ├─► Document Ingestion        — fetch Wikipedia / PMC / OpenAlex papers → Chroma vectorstore
│     ├─► Graph Indexing            — LLM extracts entities & relations per chunk → Neo4j
│     ├─► Graph Retrieval           — traverse Neo4j to collect entities + triples for the species
│     ├─► Trait Extraction          — LLM reads graph context → per-species trait list
│     └─► Hybrid Normalization      — embed traits, cluster by similarity → normalized tags + latent factors
│
├─► uPheno Enrichment (across all species at once)
│     ├─► map every extracted entity → uPheno / HP / MP ontology term
│     ├─► add ontology ancestor hierarchy into Neo4j
│     └─► compute cross-species entity similarity → DOC_SIMILAR_TO edges
│
└─► Three-Tier Cross-Species Synthesis
      ├─► Tier 1 — Ontology Convergence   (graph query, no LLM)
      ├─► Tier 2 — Semantic Similarity    (graph query, no LLM)
      └─► Tier 3 — LLM Synthesis          — merges tiers, writes gene-function hypothesis
                                             → graph_synthesis.json
```

See `docs/graph_pipeline_overview.md` for the full diagram and output file reference.

## Running it

```bash
python pipeline/run_graph_pipeline.py --species-file testcases/testcase1.json --runs 1
```

Or with a species list directly (uses GBIF for canonicalization):

```bash
python pipeline/run_graph_pipeline.py --species-list "talpa europaea,chrysochloris asiatica"
```

Batch driver scripts for running multiple testcases sequentially against a shared Chroma/PDF
cache live in `scripts/` (`run_testcases_logging.sh`, `run_testcases_chunkcap.sh`).

## Environment setup

```bash
conda env create -f SMTB2025RAG.yml
conda activate DendoLLM
```

(`SMTB2025RAG_portable.yml` is a minimal, unpinned dependency list for setting up the environment
on a different platform where the pinned conda export doesn't resolve.)

Required environment variables (shell export or `.env` in the repo root):

```bash
export OLLAMA_API_KEY="<your-open-webui-api-key>"
export OLLAMA_EMBED_MODEL="snowflake-arctic-embed2:latest"   # required — no default
export OLLAMA_CHAT_MODEL="qwen2.5:latest"                    # optional, overrides the default
export OLLAMA_BASE_URL=   # optional, this is the default
```

`OLLAMA_EMBED_MODEL` has no default — the pipeline fails without it. The Ollama backend is Open
WebUI at `dev.chat.cosy.bio`; the API key is a JWT from that site's account settings, not an
Ollama-native key. Neo4j connection settings (used by `kg/neo4j_client.py`) are also read from
the environment/`.env`.

## Repo layout

| Path | Role |
|---|---|
| `pipeline/run_graph_pipeline.py` | Orchestrator — creates a timestamped bundle dir, runs ingestion → indexing → enrichment → synthesis |
| `pipeline/graph_inventory_single.py` | Per-species wrapper: ingest → graph-index → trait extraction for one species |
| `pipeline/inventory_single_2.py` | Core RAG ingestion + MMR retrieval + LLM trait extraction (still the base extraction step under the graph pipeline) |
| `kg/graph_indexer.py` | LLM entity/relation extraction per document chunk → pushes nodes/edges into Neo4j |
| `kg/graph_retriever.py` | Traverses Neo4j to assemble the per-species subgraph context handed to the trait-extraction LLM |
| `kg/graph_enricher.py` | Maps extracted entities onto ontology terms and computes cross-species similarity edges |
| `kg/graph_synthesizer.py` | Three-tier cross-species synthesis (ontology convergence, semantic similarity, LLM synthesis) |
| `kg/ontology_index.py` | Downloads/indexes the uPheno/HP/MP ontologies and builds the embedding-based term index |
| `kg/trait_mapper.py` | Maps a free-text entity to its closest ontology term |
| `kg/kg_builder.py` | Builds the legacy Species/Trait/OntologyTerm knowledge graph from extracted traits |
| `kg/neo4j_client.py` | Neo4j driver/session helpers and schema init |
| `kg/precompute_fan_in.py` | Precomputes ontology-term fan-in counts used to score Tier-1 synthesis candidates |
| `core/rag_cli.py` | `RAG` class — Chroma vectorstore, Wikipedia/PMC/OpenAlex ingestion, MMR retrieval |
| `core/llm_backend.py` | `make_chat_llm()` / `make_embeddings()` — all LLM and embedding config lives here |
| `core/utils.py` | Shared helpers (e.g. `slugify`) |
| `analysis/analyze_runs.py` | Trait frequency and citation drift across multiple runs |
| `analysis/group_traits_llm.py` | LLM-based synonym grouping of traits → `trait_groups.json` |
| `analysis/checkChunk.py` | Debugging tool to inspect a specific retrieved chunk by its source tag |
| `scripts/build_testcase_json.py` | Builds a species JSON file from a name list, using GBIF for canonicalization |
| `scripts/run_testcases_*.sh` | Batch drivers that run several testcases sequentially against a shared cache |
| `server/` | Ollama health-check / process-status / API smoke-test helpers |
| `Prompts/` | LLM prompt templates used by the pipeline (entity extraction, graph synthesis, trait extraction, etc.) |
| `testcases/` | Example species input files, each with a known expected biological signal |
| `notebooks/` | Exploratory analysis and figure-generation notebooks |
| `docs/` | Pipeline design docs; `docs/specs/` holds historical architecture specs (not tracked in git) |

### Caching

Ingested documents, PDFs, and the Chroma vectorstore are cached locally (`pdfs/`, `chroma_store/`,
`shared_chroma/`) and are gitignored — they're rebuilt automatically on first run and reused
across subsequent runs. An `.ingest.lock` file (also gitignored, self-created) serializes
Wikipedia/PMC ingestion across concurrent runs.

### Code style

No formatter, no required type hints — match the existing style in the file you're editing.
Prefer verbose logging (more print/log output over less).
