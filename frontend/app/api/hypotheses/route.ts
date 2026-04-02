// app/api/hypotheses/route.ts
// Returns all hypotheses with their status and scores

import { NextResponse } from 'next/server'
import { runQuery } from '@/lib/neo4j'

export async function GET() {
  try {
    const records = await runQuery(`
      MATCH (h:Hypothesis)
      OPTIONAL MATCH (h)-[:DERIVED_FROM]->(p:Paper)
      WITH h, collect(DISTINCT p.title)[..3] AS source_papers
      RETURN h.id AS id,
             h.statement AS statement,
             h.status AS status,
             h.testability_score AS testability,
             h.debate_score AS debate_score,
             h.rebuttal AS rebuttal,
             source_papers
      LIMIT 50
    `)

    const hypotheses = records.map(r => ({
      id: r.id,
      statement: r.statement,
      status: r.status || 'proposed',
      testability: r.testability != null && typeof r.testability === 'object' ? r.testability.toNumber() : (r.testability ?? null),
      debate_score: r.debate_score != null && typeof r.debate_score === 'object' ? r.debate_score.toNumber() : (r.debate_score ?? null),
      rebuttal: r.rebuttal || '',
      source_papers: r.source_papers || [],
    }))

    return NextResponse.json(hypotheses)
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: 500 })
  }
}