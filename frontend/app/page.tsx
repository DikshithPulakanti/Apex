'use client'

import { useEffect, useState } from 'react'

// ── Types ────────────────────────────────────────────────────────────────

interface Stats {
  papers: number
  authors: number
  concepts: number
  hypotheses: number
  patents: number
  validated: number
  rejected: number
  relationships: number
}

interface Hypothesis {
  id: string
  statement: string
  status: string
  testability: number
  debate_score: number
  rebuttal: string
  source_papers: string[]
}

interface AgentEvent {
  type: string
  agent: string
  data: any
  timestamp: string
}

// ── Stat Card ────────────────────────────────────────────────────────────

function StatCard({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className={`rounded-xl border border-gray-800 bg-gray-900 p-6`}>
      <p className="text-sm text-gray-400">{label}</p>
      <p className={`text-3xl font-bold mt-1 ${color}`}>
        {value.toLocaleString()}
      </p>
    </div>
  )
}

// ── Status Badge ─────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    proposed: 'bg-yellow-900 text-yellow-300',
    validated: 'bg-green-900 text-green-300',
    rejected: 'bg-red-900 text-red-300',
  }
  return (
    <span className={`px-2 py-1 rounded-full text-xs font-medium ${colors[status] || 'bg-gray-700 text-gray-300'}`}>
      {status}
    </span>
  )
}

// ── Agent Icon ───────────────────────────────────────────────────────────

function AgentIcon({ agent }: { agent: string }) {
  const icons: Record<string, string> = {
    harvester: '🔬',
    reasoner: '🧠',
    skeptic: '⚔️',
    inventor: '💡',
  }
  return <span className="text-lg">{icons[agent] || '🤖'}</span>
}

// ── Main Dashboard ───────────────────────────────────────────────────────

export default function Dashboard() {
  const [stats, setStats] = useState<Stats | null>(null)
  const [hypotheses, setHypotheses] = useState<Hypothesis[]>([])
  const [events, setEvents] = useState<AgentEvent[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function fetchData() {
      try {
        const [statsRes, hypRes, eventsRes] = await Promise.all([
          fetch('/api/stats'),
          fetch('/api/hypotheses'),
          fetch('/api/events'),
        ])

        if (!statsRes.ok || !hypRes.ok || !eventsRes.ok) {
          throw new Error('API request failed')
        }

        setStats(await statsRes.json())
        setHypotheses(await hypRes.json())
        setEvents(await eventsRes.json())
      } catch (e: any) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 10000) // refresh every 10s
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <div className="text-center">
          <div className="text-4xl mb-4">🧬</div>
          <p className="text-gray-400">Loading APEX Dashboard...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-400">Error: {error}</p>
          <p className="text-gray-500 mt-2 text-sm">Make sure Docker containers are running.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <header className="border-b border-gray-800 px-8 py-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">
              <span className="text-cyan-400">APEX</span> Research Dashboard
            </h1>
            <p className="text-gray-500 text-sm mt-1">
              Autonomous Patent-Level Engineering Exchange
            </p>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
            <span className="text-gray-400 text-sm">Live</span>
          </div>
        </div>
      </header>

      <main className="px-8 py-8 space-y-8">
        {/* Stats Grid */}
        {stats && (
          <section>
            <h2 className="text-lg font-semibold text-gray-300 mb-4">Knowledge Graph</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard label="Papers" value={stats.papers} color="text-cyan-400" />
              <StatCard label="Authors" value={stats.authors} color="text-blue-400" />
              <StatCard label="Concepts" value={stats.concepts} color="text-purple-400" />
              <StatCard label="Relationships" value={stats.relationships} color="text-gray-300" />
              <StatCard label="Hypotheses" value={stats.hypotheses} color="text-yellow-400" />
              <StatCard label="Validated" value={stats.validated} color="text-green-400" />
              <StatCard label="Rejected" value={stats.rejected} color="text-red-400" />
              <StatCard label="Patents" value={stats.patents} color="text-amber-400" />
            </div>
          </section>
        )}

        {/* Two Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Hypotheses */}
          <section className="lg:col-span-2">
            <h2 className="text-lg font-semibold text-gray-300 mb-4">
              Hypotheses ({hypotheses.length})
            </h2>
            <div className="space-y-3">
              {hypotheses.map(h => (
                <div key={h.id} className="rounded-xl border border-gray-800 bg-gray-900 p-5">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <code className="text-xs text-gray-500">{h.id}</code>
                        <StatusBadge status={h.status} />
                      </div>
                      <p className="text-gray-200 text-sm leading-relaxed">
                        {h.statement}
                      </p>
                      {h.source_papers && h.source_papers.length > 0 && (
                        <div className="mt-3">
                          <p className="text-xs text-gray-500 mb-1">Source papers:</p>
                          {h.source_papers.map((p, i) => (
                            <p key={i} className="text-xs text-gray-600 truncate">• {p}</p>
                          ))}
                        </div>
                      )}
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
                    </div>
                  </div>
                </div>
              ))}
              {hypotheses.length === 0 && (
                <p className="text-gray-600 text-sm">No hypotheses yet. Run the Reasoner agent.</p>
              )}
            </div>
          </section>

          {/* Event Feed */}
          <section>
            <h2 className="text-lg font-semibold text-gray-300 mb-4">Agent Activity</h2>
            <div className="space-y-2">
              {events.map((e, i) => (
                <div key={i} className="rounded-lg border border-gray-800 bg-gray-900 p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <AgentIcon agent={e.agent} />
                    <span className="text-xs font-medium text-gray-300">{e.agent}</span>
                    <span className="text-xs text-gray-600">•</span>
                    <span className="text-xs text-gray-500">{e.type}</span>
                  </div>
                  <p className="text-xs text-gray-500 truncate">
                    {JSON.stringify(e.data).substring(0, 80)}
                  </p>
                </div>
              ))}
              {events.length === 0 && (
                <p className="text-gray-600 text-sm">No events yet.</p>
              )}
            </div>
          </section>
        </div>
      </main>
    </div>
  )
}