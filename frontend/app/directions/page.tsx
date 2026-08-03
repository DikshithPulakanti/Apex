'use client'

import { useEffect, useState } from 'react'
import HypothesisCard, { Hypothesis } from '../components/HypothesisCard'

const STATUS_FILTERS = ['all', 'proposed', 'validated', 'pending_review', 'rejected'] as const

export default function Directions() {
  const [hypotheses, setHypotheses] = useState<Hypothesis[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState<(typeof STATUS_FILTERS)[number]>('all')
  const [decidingId, setDecidingId] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)

  async function fetchHypotheses() {
    try {
      const res = await fetch('/api/hypotheses')
      if (!res.ok) throw new Error('Failed to load candidate directions')
      setHypotheses(await res.json())
      setError(null)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchHypotheses()
    const interval = setInterval(fetchHypotheses, 10000)
    return () => clearInterval(interval)
  }, [])

  async function decide(hypothesisId: string, decision: 'approved' | 'rejected') {
    setDecidingId(hypothesisId)
    setMessage(null)
    try {
      const res = await fetch(`/api/reviews/${hypothesisId}/decide`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ decision, decided_by: 'anonymous' }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Decision failed')

      setMessage(
        decision === 'approved'
          ? `Approved ${hypothesisId}${data.plan_id ? ` — research plan ${data.plan_id} drafted` : ''}`
          : `Rejected ${hypothesisId}`
      )
      await fetchHypotheses()
    } catch (e: any) {
      setMessage(`Error: ${e.message}`)
    } finally {
      setDecidingId(null)
    }
  }

  const visible = filter === 'all' ? hypotheses : hypotheses.filter(h => h.status === filter)

  return (
    <div className="min-h-screen bg-black text-white">
      <header className="border-b border-gray-800 px-8 py-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">
              <span className="text-cyan-400">APEX</span> Candidate Directions
            </h1>
            <p className="text-gray-500 text-sm mt-1">
              Every hypothesis generated so far — debate, prior art, and a suggested research
              plan where available. Scores are advisory, not a filter.
            </p>
          </div>
          <a href="/" className="text-sm text-cyan-400 hover:underline">
            ← Back to Dashboard
          </a>
        </div>
      </header>

      <main className="px-8 py-8 space-y-6 max-w-4xl mx-auto">
        <div className="flex gap-2 flex-wrap">
          {STATUS_FILTERS.map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium border ${
                filter === f
                  ? 'border-cyan-400 text-cyan-400'
                  : 'border-gray-800 text-gray-500 hover:text-gray-300'
              }`}
            >
              {f.replace('_', ' ')}
            </button>
          ))}
        </div>

        {message && (
          <div className="rounded-lg border border-gray-800 bg-gray-900 px-4 py-3 text-sm text-cyan-300">
            {message}
          </div>
        )}

        {loading && <p className="text-gray-500 text-sm">Loading candidate directions...</p>}
        {error && <p className="text-red-400 text-sm">Error: {error}</p>}

        {!loading && !error && visible.length === 0 && (
          <p className="text-gray-600 text-sm">
            No candidate directions {filter !== 'all' ? `with status "${filter}"` : 'yet'}. Run the
            Reasoner agent to generate some.
          </p>
        )}

        <div className="space-y-4">
          {visible.map(h => (
            <HypothesisCard
              key={h.id}
              hypothesis={h}
              detailed
              actions={
                h.status === 'pending_review' ? (
                  <div className="flex gap-3">
                    <button
                      onClick={() => decide(h.id, 'approved')}
                      disabled={decidingId === h.id}
                      className="px-4 py-2 rounded-lg bg-green-900 text-green-300 text-sm font-medium hover:bg-green-800 disabled:opacity-50"
                    >
                      Approve
                    </button>
                    <button
                      onClick={() => decide(h.id, 'rejected')}
                      disabled={decidingId === h.id}
                      className="px-4 py-2 rounded-lg bg-red-900 text-red-300 text-sm font-medium hover:bg-red-800 disabled:opacity-50"
                    >
                      Reject
                    </button>
                  </div>
                ) : undefined
              }
            />
          ))}
        </div>
      </main>
    </div>
  )
}
