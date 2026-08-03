// app/api/hypotheses/route.ts
// Returns all hypotheses with their status, scores, debate, prior art, and
// research plan — the full picture a student needs to compare candidate
// directions. No status filter: visibility here is unconditional, the
// automated scores are advisory context, not a gate (see /directions).

import { NextResponse } from 'next/server'
import { runQuery } from '@/lib/neo4j'

export async function GET() {
  try {
    const records = await runQuery(`
      MATCH (h:Hypothesis)
      OPTIONAL MATCH (h)-[r:CITES]->(p:Paper)
      WITH h, collect(DISTINCT {id: p.id, title: p.title, year: p.year, score: r.hybrid_score}) AS source_papers_raw
      OPTIONAL MATCH (h)-[s:SIMILAR_TO]->(sp:Paper)
      WITH h, source_papers_raw, collect(DISTINCT {id: sp.id, title: sp.title, year: sp.year, similarity: s.similarity}) AS prior_art_raw
      OPTIONAL MATCH (h)-[:HAS_PLAN]->(pl:ResearchPlan)
      RETURN h.id AS id,
             h.statement AS statement,
             h.rationale AS rationale,
             h.predicted_impact AS predicted_impact,
             h.status AS status,
             h.seed_concept AS seed_concept,
             h.testability_score AS testability,
             h.debate_score AS debate_score,
             h.rebuttal AS rebuttal,
             h.counterarguments AS counterarguments,
             h.novelty_score AS novelty_score,
             h.bert_confidence AS bert_confidence,
             h.bert_verdict AS bert_verdict,
             h.claude_score AS claude_score,
             h.claude_reasoning AS claude_reasoning,
             h.scoring_method AS scoring_method,
             [x IN source_papers_raw WHERE x.id IS NOT NULL] AS source_papers,
             [x IN prior_art_raw WHERE x.id IS NOT NULL] AS prior_art,
             CASE WHEN pl IS NULL THEN NULL ELSE {
                 methodology_sketch: pl.methodology_sketch,
                 resources_needed: pl.resources_needed,
                 first_experiment: pl.first_experiment,
                 key_related_papers: pl.key_related_papers,
                 open_risks: pl.open_risks,
                 novelty_assessment: pl.novelty_assessment
             } END AS research_plan
      ORDER BY h.id
      LIMIT 50
    `)

    const toNum = (v: any) => (v != null && typeof v === 'object' ? v.toNumber() : (v ?? null))

    const hypotheses = records.map(r => ({
      id: r.id,
      statement: r.statement,
      rationale: r.rationale || '',
      predicted_impact: r.predicted_impact || '',
      status: r.status || 'proposed',
      seed_concept: r.seed_concept || '',
      testability: toNum(r.testability),
      debate_score: toNum(r.debate_score),
      rebuttal: r.rebuttal || '',
      counterarguments: r.counterarguments || [],
      novelty_score: toNum(r.novelty_score),
      bert_confidence: toNum(r.bert_confidence),
      bert_verdict: r.bert_verdict || '',
      claude_score: toNum(r.claude_score),
      claude_reasoning: r.claude_reasoning || '',
      scoring_method: r.scoring_method || '',
      source_papers: (r.source_papers || []).map((p: any) => ({
        id: p.id, title: p.title, year: toNum(p.year), score: toNum(p.score),
      })),
      prior_art: (r.prior_art || []).map((p: any) => ({
        id: p.id, title: p.title, year: toNum(p.year), similarity: toNum(p.similarity),
      })),
      research_plan: r.research_plan ? {
        methodology_sketch: r.research_plan.methodology_sketch || '',
        resources_needed: r.research_plan.resources_needed || [],
        first_experiment: r.research_plan.first_experiment || '',
        key_related_papers: r.research_plan.key_related_papers || [],
        open_risks: r.research_plan.open_risks || [],
        novelty_assessment: r.research_plan.novelty_assessment || '',
      } : null,
    }))

    return NextResponse.json(hypotheses)
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: 500 })
  }
}
