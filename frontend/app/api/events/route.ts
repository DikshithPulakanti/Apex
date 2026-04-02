// app/api/events/route.ts
// Returns recent Kafka events (reads from a simple in-memory buffer)
// In production, this would read from a Kafka consumer or database

import { NextResponse } from 'next/server'

// In-memory event buffer (populated by SSE or polling in production)
// For now, returns mock recent events based on Neo4j state
import { runQuery } from '@/lib/neo4j'

export async function GET() {
  try {
    // Build timeline from Neo4j state
    const events: any[] = []

    const hypotheses = await runQuery(`
      MATCH (h:Hypothesis)
      RETURN h.id AS id, h.statement AS statement, h.status AS status,
             h.debate_score AS score
      ORDER BY h.created_at DESC
      LIMIT 10
    `)

    for (const h of hypotheses) {
      events.push({
        type: 'hypothesis.created',
        agent: 'reasoner',
        data: { hypothesis_id: h.id, statement: (h.statement || '').substring(0, 100) },
        timestamp: new Date().toISOString(),
      })

      if (h.status === 'validated') {
        events.push({
          type: 'hypothesis.validated',
          agent: 'skeptic',
          data: { hypothesis_id: h.id, score: h.score },
          timestamp: new Date().toISOString(),
        })
      } else if (h.status === 'rejected') {
        events.push({
          type: 'hypothesis.rejected',
          agent: 'skeptic',
          data: { hypothesis_id: h.id, score: h.score },
          timestamp: new Date().toISOString(),
        })
      }
    }

    const patents = await runQuery(`
      MATCH (p:Patent)
      RETURN p.id AS id, p.title AS title
      ORDER BY p.created_at DESC
      LIMIT 5
    `)

    for (const p of patents) {
      events.push({
        type: 'patent.drafted',
        agent: 'inventor',
        data: { patent_id: p.id, title: (p.title || '').substring(0, 100) },
        timestamp: new Date().toISOString(),
      })
    }

    return NextResponse.json(events)
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: 500 })
  }
}