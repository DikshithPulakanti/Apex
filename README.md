# APEX — Autonomous Patent-Level Engineering Exchange

> A self-evolving multi-agent AI research scientist that discovers hypotheses, debates them adversarially, and drafts patent claims — orchestrated by 4 custom MCP servers over a Neo4j knowledge graph.

## Demo

```bash
# One command: seed concept → hypothesis → debate → patent
python orchestrator.py "graph neural networks"
```

**Result:** Generates a novel research hypothesis, adversarially debates it, validates with a fine-tuned BERT model (98% F1), and drafts a patent claim — all in ~55 seconds.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Next.js Dashboard                        │
│              (live stats, hypotheses, events)                │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│                     Kafka Event Bus                         │
│  papers.ingested → hypothesis.created → hypothesis.validated │
│                                      → patent.drafted        │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│               4 LangGraph Agents                            │
│                                                             │
│  🔬 Harvester    — scrapes arXiv, builds knowledge graph    │
│  🧠 Reasoner     — finds research gaps, generates hypotheses│
│  ⚔️ Skeptic      — adversarial debate + BERT scoring        │
│  💡 Inventor     — novelty check → simulation → patent draft│
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│               4 Custom MCP Servers                          │
│                                                             │
│  paper-mcp   — search, details, concepts, neighbors         │
│  graph-mcp   — research gaps, hypotheses, graph stats       │
│  sim-mcp     — hypothesis simulation, synthetic data        │
│  patent-mcp  — prior art, novelty score, patent drafting    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│                   Data Layer                                │
│                                                             │
│  Neo4j + GDS     — knowledge graph (papers, authors,        │
│                    concepts, hypotheses, patents)            │
│  Weaviate        — vector search (semantic similarity)      │
│  PostgreSQL      — pipeline run logs                        │
│  Redis           — caching layer                            │
└─────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Framework | LangGraph + Claude API |
| MCP Servers | 4 custom servers (13 tools) |
| Knowledge Graph | Neo4j + Graph Data Science |
| Vector Search | Weaviate + all-MiniLM-L6-v2 |
| ML Model | HypothesisValidityBERT (98% F1) — [HuggingFace](https://huggingface.co/Dikshith4500/HypothesisValidityBERT) |
| Event Streaming | Apache Kafka |
| Frontend | Next.js + Tailwind CSS |
| Databases | PostgreSQL + Redis |
| Infrastructure | Docker Compose (8 services) |
| ML Ops | MLflow + DVC |

## Knowledge Graph Stats

- **100+ Paper nodes** across 20 arXiv categories
- **368 Author nodes** linked to papers
- **780 Concept nodes** extracted and embedded
- **6 Hypotheses** generated and debated
- **2 Patents** drafted autonomously
- **3,984 Relationships** connecting the graph

## HypothesisValidityBERT

Fine-tuned BERT model for scoring scientific hypothesis validity. Trained on 2,600 synthetic examples (valid + 6 flaw types) across 20 research domains.

| Metric | Score |
|--------|-------|
| Accuracy | 98.08% |
| F1 | 0.9821 |
| Precision | 0.9856 |
| Recall | 0.9786 |

Integrated into the Skeptic agent as a two-stage scorer: BERT handles high-confidence cases instantly (free), Claude judges uncertain cases. Reduces API costs by ~30%.

Model: [huggingface.co/Dikshith4500/HypothesisValidityBERT](https://huggingface.co/Dikshith4500/HypothesisValidityBERT)

## Quick Start

```bash
# Clone
git clone https://github.com/DikshithPulakanti/Apex.git
cd Apex

# Start all 8 services
docker-compose up -d

# Create virtual environment
python3.11 -m venv apex_env311
source apex_env311/bin/activate
pip install -r requirements.txt

# Run the full pipeline
python orchestrator.py "reinforcement learning"

# Start the dashboard
cd frontend && npm install && npm run dev
# Open http://localhost:3000
```

## Project Structure

```
apex/
├── agents/
│   ├── harvester.py          # arXiv scraper agent
│   ├── reasoner.py           # hypothesis generation agent
│   ├── skeptic.py            # adversarial debate agent (BERT + Claude)
│   └── inventor.py           # patent drafting agent
├── database/
│   ├── neo4j_client.py       # Neo4j graph database client
│   ├── postgres_client.py    # PostgreSQL operational logger
│   ├── redis_client.py       # Redis caching layer
│   ├── weaviate_client.py    # Weaviate vector search client
│   ├── embedder.py           # Sentence transformer embeddings
│   └── schema.py             # Graph schema + agent node setup
├── events/
│   ├── kafka_manager.py      # Kafka producer/consumer/topics
│   └── agent_events.py       # Agent event emission helpers
├── mcp_servers/
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
│   └── ingest.py             # Full ingestion pipeline
├── scrapers/
│   ├── arxiv_scraper.py      # Async arXiv paper scraper
│   └── queries.py            # 20 APEX search queries
├── frontend/                 # Next.js dashboard
│   ├── app/
│   │   ├── page.tsx          # Main dashboard
│   │   └── api/              # Stats, hypotheses, events endpoints
│   └── lib/neo4j.ts          # Neo4j driver for API routes
├── orchestrator.py           # Full pipeline: seed → patent
├── docker-compose.yml        # 8 services (Neo4j, Postgres, Redis,
│                             #   Weaviate, Kafka, Zookeeper, app)
└── README.md
```

## Agents

### 🔬 Harvester
Scrapes arXiv across 20 categories using async aiohttp. Deduplicates, batches into Neo4j, builds author graph. Rate-limited with exponential backoff.

### 🧠 Reasoner
Queries Neo4j GDS for research gaps (concept pairs with high distance, low direct connections). Gathers context papers from Weaviate via semantic search. Sends gap + papers to Claude to generate novel, testable hypotheses.

### ⚔️ Skeptic
Three-phase adversarial debate: Claude generates counterarguments → Claude generates rebuttal → Two-stage scoring (BERT first, Claude fallback). Only hypotheses scoring ≥ 0.6 are validated.

### 💡 Inventor
Checks novelty against existing graph. Runs hypothesis simulation. If novelty + simulation pass threshold, Claude drafts patent claims (title, abstract, independent claim, dependent claims). Stores Patent node in Neo4j linked to source hypothesis.

## License

MIT