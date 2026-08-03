// app/components/HypothesisCard.tsx
// Shared hypothesis card — used by the dashboard's summary list (compact)
// and /directions' browse view (detailed), so the two don't duplicate
// rendering logic for the same data.

export interface SourcePaper {
  id: string
  title: string
  year: number | null
  score: number | null
}

export interface PriorArtPaper {
  id: string
  title: string
  year: number | null
  similarity: number | null
}

export interface ResearchPlan {
  methodology_sketch: string
  resources_needed: string[]
  first_experiment: string
  key_related_papers: string[]
  open_risks: string[]
  novelty_assessment: string
}

export interface Hypothesis {
  id: string
  statement: string
  rationale?: string
  predicted_impact?: string
  status: string
  seed_concept?: string
  testability: number | null
  debate_score: number | null
  rebuttal: string
  counterarguments?: string[]
  novelty_score?: number | null
  bert_confidence?: number | null
  bert_verdict?: string
  claude_score?: number | null
  claude_reasoning?: string
  scoring_method?: string
  source_papers: SourcePaper[]
  prior_art?: PriorArtPaper[]
  research_plan?: ResearchPlan | null
}

const STATUS_STYLES: Record<string, string> = {
  validated: 'bg-green-900 text-green-300',
  rejected: 'bg-red-900 text-red-300',
  pending_review: 'bg-yellow-900 text-yellow-300',
  proposed: 'bg-gray-800 text-gray-400',
}

export function StatusBadge({ status }: { status: string }) {
  return (
    <span className={`px-2 py-1 rounded-full text-xs font-medium ${STATUS_STYLES[status] || STATUS_STYLES.proposed}`}>
      {status.replace('_', ' ')}
    </span>
  )
}

function PaperList({ title, papers }: {
  title: string
  papers: { id: string; title: string; year: number | null; extra?: string }[]
}) {
  if (papers.length === 0) return null
  return (
    <div className="mt-3">
      <p className="text-xs text-gray-500 mb-1">{title}:</p>
      {papers.map((p, i) => (
        <p key={i} className="text-xs text-gray-600 truncate">
          •{' '}
          {p.id ? (
            <a href={`https://arxiv.org/abs/${p.id}`} target="_blank" rel="noopener noreferrer"
               className="hover:text-cyan-400 hover:underline">
              {p.title}
            </a>
          ) : p.title}
          {p.year ? ` (${p.year})` : ''}
          {p.extra ? ` — ${p.extra}` : ''}
        </p>
      ))}
    </div>
  )
}

export default function HypothesisCard({
  hypothesis: h, detailed = false, actions,
}: {
  hypothesis: Hypothesis
  detailed?: boolean
  actions?: React.ReactNode
}) {
  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900 p-5">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-3 mb-2 flex-wrap">
            <code className="text-xs text-gray-500">{h.id}</code>
            <StatusBadge status={h.status} />
            {h.seed_concept && <span className="text-xs text-gray-600">from "{h.seed_concept}"</span>}
          </div>

          <p className="text-gray-200 text-sm leading-relaxed">{h.statement}</p>

          {detailed && h.rationale && (
            <p className="text-xs text-gray-500 mt-2 leading-relaxed">
              <span className="text-gray-400">Rationale: </span>{h.rationale}
            </p>
          )}
          {detailed && h.predicted_impact && (
            <p className="text-xs text-gray-500 mt-1 leading-relaxed">
              <span className="text-gray-400">If validated: </span>{h.predicted_impact}
            </p>
          )}

          <PaperList
            title="Source papers"
            papers={h.source_papers.map(p => ({ id: p.id, title: p.title, year: p.year }))}
          />

          {h.rebuttal && (
            <div className="mt-3">
              {h.counterarguments && h.counterarguments.length > 0 && (
                <p className="text-xs text-gray-500 mb-1">
                  {h.counterarguments.length} counterargument(s) raised during debate
                </p>
              )}
              {detailed && h.counterarguments && h.counterarguments.length > 0 && (
                <ul className="list-disc list-inside text-xs text-gray-500 mb-2 space-y-0.5">
                  {h.counterarguments.map((c, i) => <li key={i}>{c}</li>)}
                </ul>
              )}
              <p className="text-xs text-gray-500 mb-1">Rebuttal:</p>
              <p className="text-xs text-gray-400 leading-relaxed">{h.rebuttal}</p>
            </div>
          )}

          {detailed && h.prior_art && (
            <PaperList
              title="Closest existing work (prior art)"
              papers={h.prior_art.map(p => ({
                id: p.id, title: p.title, year: p.year,
                extra: p.similarity != null ? `similarity ${p.similarity.toFixed(2)}` : undefined,
              }))}
            />
          )}

          {detailed && h.research_plan && (
            <div className="mt-4 rounded-lg border border-gray-800 p-3 space-y-2">
              <p className="text-xs font-medium text-amber-400">Suggested next steps</p>
              <p className="text-xs text-gray-400">{h.research_plan.methodology_sketch}</p>
              <p className="text-xs text-gray-500">
                <span className="text-gray-400">First experiment: </span>
                {h.research_plan.first_experiment}
              </p>
              {h.research_plan.resources_needed.length > 0 && (
                <p className="text-xs text-gray-500">
                  <span className="text-gray-400">Resources needed: </span>
                  {h.research_plan.resources_needed.join(', ')}
                </p>
              )}
              {h.research_plan.open_risks.length > 0 && (
                <p className="text-xs text-gray-500">
                  <span className="text-gray-400">Open risks: </span>
                  {h.research_plan.open_risks.join('; ')}
                </p>
              )}
              <p className="text-xs text-gray-500">
                <span className="text-gray-400">Novelty assessment: </span>
                {h.research_plan.novelty_assessment}
              </p>
            </div>
          )}

          {actions && <div className="mt-4">{actions}</div>}
        </div>

        <div className="text-right shrink-0">
          {h.testability != null && (
            <p className="text-xs text-gray-500">
              Testability: <span className="text-cyan-400">{h.testability}</span>
            </p>
          )}
          {h.debate_score != null && (
            <p className="text-xs text-gray-500">
              Debate: <span className="text-yellow-400">{h.debate_score}</span>
            </p>
          )}
          {detailed && h.novelty_score != null && (
            <p className="text-xs text-gray-500">
              Novelty: <span className="text-purple-400">{h.novelty_score}</span>
            </p>
          )}
          {detailed && h.scoring_method && (
            <p className="text-xs text-gray-600 mt-1">{h.scoring_method.replace('_', ' ')}</p>
          )}
        </div>
      </div>
    </div>
  )
}
