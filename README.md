# APEX — Autonomous Patent-Level Engineering Exchange

> A self-evolving multi-agent AI research scientist that discovers hypotheses, debates them adversarially, and drafts patent claims — orchestrated by 4 custom MCP servers over a Neo4j knowledge graph, with confidence-gated human review for anything neither model is sure about.

## Demo

```bash
# One command: seed concept → hypothesis → debate → patent (or human review)
python orchestrator.py "graph neural networks"
```

**Result:** Generates a novel research hypothesis, adversarially debates it, scores it with a fine-tuned BERT model — falling back to a Claude judge when BERT is unsure, and to a human reviewer when neither is — then drafts a patent claim for anything that passes novelty + simulation checks.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│      Next.js Dashboard  +  /review Human Review Queue       │
└───────────────┬───────────────────────────┬─────────────────┘
                │                           │
┌───────────────┴───────────────┐  ┌────────┴──────────────────┐
│         bridge.py             │  │  Postgres: hypothesis_     │
│  real Kafka consumer + the    │  │  reviews queue (enqueue /  │
│  human-review decide API      │  │  get_pending / decide)     │
└───────────────┬───────────────┘  └────────┬──────────────────┘
                │                            │
┌───────────────┴────────────────────────────┴─────────────────┐
│                     Kafka Event Bus (6 topics)                │
│  papers.ingested → hypothesis.created → hypothesis.validated/ │
│  rejected → patent.drafted → agent.status (per-node detail)   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│               4 LangGraph Agents (18 traced nodes)           │
│                                                             │
│  Harvester    — scrapes arXiv, builds knowledge graph        │
│  Reasoner     — finds research gaps, generates hypotheses    │
│  Skeptic      — adversarial debate + 3-tier confidence gate  │
│  Inventor     — novelty check → simulation → patent draft    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│      4 Custom MCP Servers — independent, typed, Dockerized   │
│      (stdio for local clients, streamable HTTP as services)  │
│                                                             │
│  paper-mcp   — search, details, concepts, neighbors          │
│  graph-mcp   — research gaps, hypotheses, graph stats        │
│  sim-mcp     — hypothesis simulation, synthetic data          │
│  patent-mcp  — prior art, novelty score, patent drafting      │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│                   Data + MLOps Layer                          │
│                                                             │
│  Neo4j + GDS     — knowledge graph (PageRank, Louvain,        │
│                    Betweenness over the concept graph)         │
│  Weaviate        — hybrid dense + BM25 search                 │
│  PostgreSQL      — pipeline run logs + human review queue     │
│  Redis           — caching layer                              │
│  MLflow          — BERT training runs + one run per Skeptic   │
│                    debate decision (skeptic-evaluations)       │
│  DVC             — versions the corpus snapshot, training      │
│                    data, and model checkpoint                  │
└─────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Framework | LangGraph + Claude API (4 agents, 18 traced nodes) |
| MCP Servers | 4 independent Docker services, stdio + streamable HTTP, Pydantic-typed inputs |
| Knowledge Graph | Neo4j + Graph Data Science (PageRank, Louvain, Betweenness) |
| Vector Search | Weaviate hybrid (dense + BM25) + all-MiniLM-L6-v2 |
| ML Model | HypothesisValidityBERT (98% F1) — [HuggingFace](https://huggingface.co/Dikshith4500/HypothesisValidityBERT) |
| Event Streaming | Apache Kafka — 6 topics, per-node tracing via `bridge.py` |
| Human Review | Postgres-backed queue + Next.js `/review` UI, gated by a 3-tier confidence check |
| Frontend | Next.js + Tailwind CSS |
| Databases | PostgreSQL + Redis |
| Infrastructure | Docker Compose (11 services) |
| ML Ops | MLflow (training + per-decision runs) + DVC (5-stage pipeline) |
| CI/CD | GitHub Actions — mocked unit job on every PR, live-infra integration job on `main` |

## Knowledge Graph Stats

*(measured directly from the running database, not aspirational)*

- **9,851 Paper nodes** scraped from arXiv across 28 categories/cross-domain queries
- **30,170 Author nodes** linked to papers
- **6,704 Concept nodes** extracted and scored with PageRank/Louvain/Betweenness (from the first processed tranche of ~1,490 papers — concept extraction for the rest is a separate, LLM-cost-bound step you run via `python pipeline/extract_concepts.py`)
- **13 Hypotheses** generated and debated, **5 Patents** drafted autonomously
- **56,284 Relationships** connecting the graph

## Confidence-Gated Evaluation

The Skeptic agent scores every hypothesis through three tiers, each logged as its own MLflow run under the `skeptic-evaluations` experiment:

1. **BERT confident (≥ 0.95)** — auto-decided instantly, for free.
2. **BERT unsure, Claude decisive** (score ≥ 0.70 agreeing "approved", or ≤ 0.30 agreeing "rejected") — still fully automated.
3. **Neither is confident** (or Claude's call fails) — flagged `pending_review` in a Postgres queue instead of guessing, surfaced on the `/review` dashboard page with both models' scores and reasoning. A human's Approve/Reject decision updates Neo4j, publishes to Kafka, and — on approval — triggers the Inventor agent the same way an auto-approval would.

## HypothesisValidityBERT

Fine-tuned BERT model for scoring scientific hypothesis validity. Trained on 2,600 synthetic examples (valid + 6 flaw types) across 20 research domains.

| Metric | Score |
|--------|-------|
| Accuracy | 98.08% |
| F1 | 0.9821 |
| Precision | 0.9856 |
| Recall | 0.9786 |

Model: [huggingface.co/Dikshith4500/HypothesisValidityBERT](https://huggingface.co/Dikshith4500/HypothesisValidityBERT)

## Testing & CI/CD

- **56 unit tests**, fully mocked (no live infra, no Claude API spend) — covering all 4 agents' node logic, all 4 MCP servers' tool handlers, the Kafka publisher, the ingestion pipeline's branches, and the BERT predictor.
- **3 integration tests**, marked `@pytest.mark.integration`, exercising the real arXiv → Neo4j → Redis pipeline and the Weaviate/GDS semantic layer against live infra.
- `.github/workflows/ci.yml` runs the unit suite on every push/PR, and the integration suite against a real `docker-compose` stack on pushes to `main` or manual dispatch.

```bash
pytest -m "not integration"   # fast, mocked, what CI runs on every PR
pytest -m integration         # needs docker-compose up first
```

## Reproducibility (DVC + MLflow)

```bash
dvc dag              # ingest → extract_concepts → run_gds → export_corpus_snapshot / train_bert
dvc repro             # re-run any stage; train_bert has real, cached file deps/outs
dvc push              # send tracked artifacts (corpus snapshot, training data, model) to the remote
```

Every Skeptic decision's MLflow run ID is stored on both the Neo4j hypothesis node and the Postgres review row, so any verdict — auto or human-reviewed — traces back to exactly what BERT and Claude saw.

## Quick Start

```bash
# Clone
git clone https://github.com/DikshithPulakanti/Apex.git
cd Apex

# Start infra + all 4 MCP servers as independent Docker services
docker compose up -d

# Create virtual environment
python3.11 -m venv apex_env311
source apex_env311/bin/activate
pip install -r requirements-dev.txt   # includes requirements.txt + pytest tooling

# Run the full pipeline
python orchestrator.py "reinforcement learning"

# Start the event/review bridge (needed for real-time events + the review queue)
python bridge.py

# Start the dashboard
cd frontend && npm install && npm run dev
# Open http://localhost:3000 (dashboard) and http://localhost:3000/review (review queue)
```

Ports in `docker-compose.yml` are shifted off Neo4j/Weaviate/Kafka's usual defaults (7688, 8081, 9093, ...) — that's a local accommodation for other projects sharing this machine, not a requirement; change them freely if you don't need to.

## Project Structure

```
apex/
├── agents/
│   ├── harvester.py          # arXiv scraper agent
│   ├── reasoner.py           # hypothesis generation agent
│   ├── skeptic.py            # adversarial debate + 3-tier confidence gate
│   └── inventor.py           # patent drafting agent
├── database/
│   ├── neo4j_client.py       # Neo4j graph database client
│   ├── postgres_client.py    # pipeline run logs + human review queue
│   ├── redis_client.py       # Redis caching layer
│   ├── weaviate_client.py    # Weaviate hybrid search client
│   ├── embedder.py           # Sentence transformer embeddings
│   └── schema.py             # Graph schema + agent node setup
├── events/
│   ├── kafka_manager.py      # Kafka producer/consumer/topics
│   ├── agent_events.py       # Agent event emission helpers
│   └── node_tracing.py       # per-node start/complete/error tracing
├── mcp_servers/
│   ├── _transport.py         # shared stdio / streamable-HTTP transport
│   ├── schemas.py            # Pydantic input models (typed validation)
│   ├── Dockerfile            # shared image, parameterized per server
│   ├── paper_mcp.py          # Paper search + details (5 tools)
│   ├── graph_mcp.py          # Research gaps + hypotheses (5 tools)
│   ├── sim_mcp.py            # Simulation + synthetic data (3 tools)
│   └── patent_mcp.py         # Prior art + patent drafting (3 tools)
├── training/
│   ├── generate_dataset.py   # Synthetic hypothesis dataset generator
│   ├── train_bert.py         # BERT fine-tuning with MLflow
│   ├── predictor.py          # Inference wrapper
│   └── push_to_hub.py        # HuggingFace upload
├── pipeline/
│   ├── ingest.py              # Full ingestion pipeline
│   ├── extract_concepts.py    # Claude-based concept extraction
│   ├── run_gds.py             # PageRank / Louvain / Betweenness
│   └── export_corpus_snapshot.py  # DVC-tracked corpus export
├── scrapers/
│   ├── arxiv_scraper.py      # Async arXiv paper scraper
│   └── queries.py            # 28 APEX search queries
├── tests/                     # 56 unit tests + tests/integration/
├── frontend/                  # Next.js dashboard + /review queue
│   ├── app/
│   │   ├── page.tsx          # Main dashboard
│   │   ├── review/page.tsx   # Human review queue
│   │   └── api/              # Stats, hypotheses, events, reviews endpoints
│   └── lib/neo4j.ts           # Neo4j driver for API routes
├── orchestrator.py            # Full pipeline: seed → patent or pending_review
├── bridge.py                  # Real Kafka consumer + review-decide API (runs on the host, not in Compose)
├── dvc.yaml                   # 5-stage reproducible pipeline
├── docker-compose.yml         # 11 services: Neo4j, Postgres, Redis, Weaviate,
│                              #   Kafka, Zookeeper, app, + 4 MCP servers
└── README.md
```

## Agents

### Harvester
Scrapes arXiv across 28 categories/cross-domain queries using async aiohttp, rate-limited to stay well within arXiv's usage policy. Deduplicates via Redis, batches into Neo4j, builds the author graph.

### Reasoner
Queries Neo4j GDS for research gaps (cross-community concept pairs with high PageRank but low co-occurrence). Gathers context papers from Weaviate via hybrid search. Sends the gap + papers to Claude to generate a novel, testable hypothesis.

### Skeptic
Three-phase adversarial debate: Claude generates counterarguments → Claude generates a rebuttal → three-tier confidence-gated scoring (BERT → Claude judge → human review). Every scoring decision is logged to MLflow.

### Inventor
Checks novelty against the existing graph, runs a Monte Carlo simulation, and only drafts a patent (title, abstract, independent + dependent claims) if both pass threshold. Stores the Patent node in Neo4j linked to its source hypothesis.

## License

MIT
