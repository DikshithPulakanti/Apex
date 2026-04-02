// app/api/graph/route.ts
// Returns nodes and edges for the knowledge graph visualization

import { NextResponse } from 'next/server'
import { runQuery } from '@/lib/neo4j'

export async function GET() {
  try {
    // Get hypotheses and their connections
    const hypotheses = await runQuery(`
      MATCH (h:Hypothesis)
      RETURN h.id AS id, h.statement AS statement, h.status AS status,
             h.debate_score AS debate_score, 'hypothesis' AS type
    `)

    // Get patents and their source hypotheses
    const patents = await runQuery(`
      MATCH (p:Patent)
      OPTIONAL MATCH (p)-[:DERIVED_FROM]->(h:Hypothesis)
      RETURN p.id AS id, p.title AS title, 'patent' AS type,
             h.id AS hypothesis_id
    `)

    // Get top concepts connected to hypotheses
    const concepts = await runQuery(`
      MATCH (c:Concept)
      WITH c, COUNT { (c)--() } AS connections
      ORDER BY connections DESC
      LIMIT 30
      RETURN c.name AS id, c.name AS name, 'concept' AS type, connections
    `)

    // Get agents
    const agents = await runQuery(`
      MATCH (a:Agent)
      RETURN a.name AS id, a.name AS name, 'agent' AS type
    `)

    // Get sample papers (top connected)
    const papers = await runQuery(`
      MATCH (p:Paper)
      WITH p, COUNT { (p)--() } AS connections
      ORDER BY connections DESC
      LIMIT 20
      RETURN p.id AS id, p.title AS title, 'paper' AS type, connections
    `)

    // Get edges between concepts
    const conceptEdges = await runQuery(`
      MATCH (c1:Concept)-[r]-(c2:Concept)
      WITH c1, c2, type(r) AS rel
      WHERE c1.name < c2.name
      WITH c1.name AS source, c2.name AS target, rel
      LIMIT 50
      RETURN source, target, rel AS type
    `)

    // Get paper-concept edges
    const paperConceptEdges = await runQuery(`
      MATCH (p:Paper)-[:HAS_CONCEPT]->(c:Concept)
      WITH p, c, COUNT { (c)--() } AS cConn
      ORDER BY cConn DESC
      WITH p.id AS source, c.name AS target
      LIMIT 40
      RETURN source, target, 'HAS_CONCEPT' AS type
    `)

    // Build nodes
    const nodes: any[] = []

    for (const h of hypotheses) {
      nodes.push({
        data: {
          id: h.id,
          label: (h.statement || '').substring(0, 50) + '...',
          type: 'hypothesis',
          status: h.status || 'proposed',
          score: h.debate_score != null && typeof h.debate_score === 'object'
            ? h.debate_score.toNumber() : (h.debate_score ?? null),
        }
      })
    }

    for (const p of patents) {
      nodes.push({
        data: {
          id: p.id,
          label: (p.title || '').substring(0, 50) + '...',
          type: 'patent',
          hypothesis_id: p.hypothesis_id,
        }
      })
    }

    for (const c of concepts) {
      nodes.push({
        data: {
          id: `concept_${c.id}`,
          label: c.name,
          type: 'concept',
          connections: typeof c.connections === 'object' ? c.connections.toNumber() : c.connections,
        }
      })
    }

    for (const a of agents) {
      nodes.push({
        data: {
          id: `agent_${a.id}`,
          label: a.name,
          type: 'agent',
        }
      })
    }

    for (const p of papers) {
      nodes.push({
        data: {
          id: p.id,
          label: (p.title || '').substring(0, 40) + '...',
          type: 'paper',
          connections: typeof p.connections === 'object' ? p.connections.toNumber() : p.connections,
        }
      })
    }

    // Build edges
    const edges: any[] = []

    // Patent → Hypothesis
    for (const p of patents) {
      if (p.hypothesis_id) {
        edges.push({
          data: { source: p.id, target: p.hypothesis_id, type: 'DERIVED_FROM' }
        })
      }
    }

    // Concept ↔ Concept
    for (const e of conceptEdges) {
      edges.push({
        data: { source: `concept_${e.source}`, target: `concept_${e.target}`, type: e.type }
      })
    }

    // Paper → Concept
    for (const e of paperConceptEdges) {
      edges.push({
        data: { source: e.source, target: `concept_${e.target}`, type: 'HAS_CONCEPT' }
      })
    }

    return NextResponse.json({ nodes, edges })
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: 500 })
  }
}