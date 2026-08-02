'use client'

import { useEffect, useState } from 'react'

interface PendingReview {
  id: number
  hypothesis_id: string
  statement_snapshot: string
  bert_confidence: number
  bert_verdict: string
  claude_score: number
  claude_verdict: string
  claude_reasoning: string
  mlflow_run_id: string
  status: string
  created_at: string
}

export default function ReviewQueue() {
  const [reviews, setReviews] = useState<PendingReview[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [notesById, setNotesById] = useState<Record<string, string>>({})
  const [reviewerName, setReviewerName] = useState('')
  const [decidingId, setDecidingId] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)

  async function fetchReviews() {
    try {
      const res = await fetch('/api/reviews')
      if (!res.ok) throw new Error('Failed to load review queue')
      setReviews(await res.json())
      setError(null)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchReviews()
    const interval = setInterval(fetchReviews, 10000)
    return () => clearInterval(interval)
  }, [])

  async function decide(hypothesisId: string, decision: 'approved' | 'rejected') {
    setDecidingId(hypothesisId)
    setMessage(null)
    try {
      const res = await fetch(`/api/reviews/${hypothesisId}/decide`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          decision,
          decided_by: reviewerName || 'anonymous',
          decision_notes: notesById[hypothesisId] || '',
        }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Decision failed')

      setMessage(
        decision === 'approved'
          ? `Approved ${hypothesisId}${data.patent_id ? ` — patent ${data.patent_id} drafted` : ''}`
          : `Rejected ${hypothesisId}`
      )
      await fetchReviews()
    } catch (e: any) {
      setMessage(`Error: ${e.message}`)
    } finally {
      setDecidingId(null)
    }
  }

  return (
    <div className="min-h-screen bg-black text-white">
      <header className="border-b border-gray-800 px-8 py-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">
              <span className="text-cyan-400">APEX</span> Human Review Queue
            </h1>
            <p className="text-gray-500 text-sm mt-1">
              Hypotheses neither BERT nor Claude could confidently auto-decide
            </p>
          </div>
          <a href="/" className="text-sm text-cyan-400 hover:underline">
            ← Back to Dashboard
          </a>
        </div>
      </header>

      <main className="px-8 py-8 space-y-6 max-w-4xl mx-auto">
        <div>
          <label className="text-xs text-gray-500 block mb-1">Reviewer name</label>
          <input
            type="text"
            value={reviewerName}
            onChange={e => setReviewerName(e.target.value)}
            placeholder="your name"
            className="bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-sm w-64"
          />
        </div>

        {message && (
          <div className="rounded-lg border border-gray-800 bg-gray-900 px-4 py-3 text-sm text-cyan-300">
            {message}
          </div>
        )}

        {loading && <p className="text-gray-500 text-sm">Loading review queue...</p>}
        {error && <p className="text-red-400 text-sm">Error: {error}</p>}

        {!loading && !error && reviews.length === 0 && (
          <p className="text-gray-600 text-sm">
            Nothing pending — every hypothesis so far was decisive enough for BERT or Claude to auto-decide.
          </p>
        )}

        <div className="space-y-4">
          {reviews.map(r => (
            <div key={r.hypothesis_id} className="rounded-xl border border-gray-800 bg-gray-900 p-6">
              <div className="flex items-center gap-3 mb-3">
                <code className="text-xs text-gray-500">{r.hypothesis_id}</code>
                <span className="px-2 py-1 rounded-full text-xs font-medium bg-yellow-900 text-yellow-300">
                  pending review
                </span>
              </div>

              <p className="text-gray-200 text-sm leading-relaxed mb-4">{r.statement_snapshot}</p>

              <div className="grid grid-cols-2 gap-4 mb-4 text-xs">
                <div className="rounded-lg border border-gray-800 p-3">
                  <p className="text-gray-500 mb-1">BERT (fast-pass model)</p>
                  <p className="text-gray-200">
                    {r.bert_verdict} — confidence {r.bert_confidence?.toFixed(2)}
                  </p>
                </div>
                <div className="rounded-lg border border-gray-800 p-3">
                  <p className="text-gray-500 mb-1">Claude Judge</p>
                  <p className="text-gray-200">
                    {r.claude_verdict} — score {r.claude_score?.toFixed(2)}
                  </p>
                </div>
              </div>

              {r.claude_reasoning && (
                <p className="text-xs text-gray-500 mb-4">
                  <span className="text-gray-400">Claude's reasoning: </span>
                  {r.claude_reasoning}
                </p>
              )}

              {r.mlflow_run_id && (
                <p className="text-xs text-gray-600 mb-4">MLflow run: {r.mlflow_run_id}</p>
              )}

              <textarea
                value={notesById[r.hypothesis_id] || ''}
                onChange={e =>
                  setNotesById(prev => ({ ...prev, [r.hypothesis_id]: e.target.value }))
                }
                placeholder="Optional notes explaining your decision..."
                className="w-full bg-black border border-gray-800 rounded-lg px-3 py-2 text-sm mb-4"
                rows={2}
              />

              <div className="flex gap-3">
                <button
                  onClick={() => decide(r.hypothesis_id, 'approved')}
                  disabled={decidingId === r.hypothesis_id}
                  className="px-4 py-2 rounded-lg bg-green-900 text-green-300 text-sm font-medium hover:bg-green-800 disabled:opacity-50"
                >
                  Approve
                </button>
                <button
                  onClick={() => decide(r.hypothesis_id, 'rejected')}
                  disabled={decidingId === r.hypothesis_id}
                  className="px-4 py-2 rounded-lg bg-red-900 text-red-300 text-sm font-medium hover:bg-red-800 disabled:opacity-50"
                >
                  Reject
                </button>
              </div>
            </div>
          ))}
        </div>
      </main>
    </div>
  )
}
