// app/api/stats/route.ts
// Returns knowledge graph statistics

import { NextResponse } from 'next/server'
import { runQuery } from '@/lib/neo4j'

export async function GET() {
  try {
    const [papers] = await runQuery('MATCH (p:Paper) RETURN count(p) AS count')
    const [authors] = await runQuery('MATCH (a:Author) RETURN count(a) AS count')
    const [concepts] = await runQuery('MATCH (c:Concept) RETURN count(c) AS count')
    const [hypotheses] = await runQuery('MATCH (h:Hypothesis) RETURN count(h) AS count')
    const [patents] = await runQuery('MATCH (p:Patent) RETURN count(p) AS count')
    const [validated] = await runQuery("MATCH (h:Hypothesis {status: 'validated'}) RETURN count(h) AS count")
    const [rejected] = await runQuery("MATCH (h:Hypothesis {status: 'rejected'}) RETURN count(h) AS count")
    const [relationships] = await runQuery('MATCH ()-[r]->() RETURN count(r) AS count')

    return NextResponse.json({
      papers: (papers.count as any).toNumber?.() ?? papers.count,
      authors: (authors.count as any).toNumber?.() ?? authors.count,
      concepts: (concepts.count as any).toNumber?.() ?? concepts.count,
      hypotheses: (hypotheses.count as any).toNumber?.() ?? hypotheses.count,
      patents: (patents.count as any).toNumber?.() ?? patents.count,
      validated: (validated.count as any).toNumber?.() ?? validated.count,
      rejected: (rejected.count as any).toNumber?.() ?? rejected.count,
      relationships: (relationships.count as any).toNumber?.() ?? relationships.count,
    })
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: 500 })
  }
}