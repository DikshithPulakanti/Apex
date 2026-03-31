# APEX — Autonomous Patent-Level Engineering Exchange

> A self-evolving multi-agent AI research scientist that discovers hypotheses, debates them adversarially, and drafts patent claims — orchestrated by 4 custom MCP servers over a Neo4j knowledge graph.

---

## Current Status
Week 1 complete — 1,500+ papers ingested across 20 arXiv categories.

---

## Architecture
```
Layer 7: Frontend (Next.js + D3 + Cytoscape.js)
Layer 6: API Gateway (FastAPI + GraphQL)
Layer 5: MCP Orchestration (4 custom MCP servers)
Layer 4: Multi-Agent Society (LangGraph — Harvester, Reasoner, Skeptic, Inventor)
Layer 3: Knowledge Graph (Neo4j + Weaviate)
Layer 2: Data Pipeline (arXiv scraper + embeddings)
Layer 1: Infrastructure (Docker + Kafka + PostgreSQL + Redis)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Knowledge Graph | Neo4j + GDS |
| Agent Framework | LangGraph + Claude API |
| MCP Servers | Custom MCP (paper, graph, sim, patent) |
| Vector Search | Weaviate |
| Async Scraping | aiohttp + asyncio |
| Databases | PostgreSQL + Redis |
| Infrastructure | Docker + Kafka |
| Frontend | Next.js + D3.js |

---

## Quick Start
```bash
# Clone the repo
git clone https://github.com/DikshithPulakanti/Apex.git
cd Apex

# Start all services
docker-compose up -d

# Run the ingestion pipeline
python pipeline/ingest.py
```

---

## Project Structure
```
apex/
├── database/
│   ├── neo4j_client.py      # Neo4j graph database client
│   ├── postgres_client.py   # PostgreSQL operational logger
│   ├── redis_client.py      # Redis caching layer
│   └── schema.py            # Graph schema + agent node setup
├── scrapers/
│   ├── arxiv_scraper.py     # Async arXiv paper scraper
│   └── queries.py           # 20 APEX search queries
├── pipeline/
│   └── ingest.py            # Full ingestion pipeline
├── tests/
│   └── test_pipeline.py     # Integration tests
├── docker-compose.yml       # Neo4j + PostgreSQL + Redis
└── README.md
```

---

## Knowledge Graph Stats
- **1,500+ Paper nodes** across 20 arXiv categories
- **5,000+ Author nodes** linked to their papers
- **5,300+ Relationships** connecting the graph
- **4 Agent nodes** — Harvester, Reasoner, Skeptic, Inventor

---

## Roadmap

- [x] Week 1 — Infrastructure + Data Pipeline
- [ ] Week 2 — Embeddings + Weaviate + Neo4j GDS
- [ ] Week 3 — LangGraph Agents (Harvester + Reasoner)
- [ ] Week 4 — Custom MCP Servers
- [ ] Week 5 — Skeptic + Inventor Agents
- [ ] Week 6 — Synthetic Dataset + HuggingFace
- [ ] Week 7 — Kafka + Next.js Frontend
- [ ] Week 8 — Polish + Demo + Launch

---

*Built from scratch. Patent-level. Graduation portfolio project.*