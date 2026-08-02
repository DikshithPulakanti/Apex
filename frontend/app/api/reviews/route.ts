// app/api/reviews/route.ts
// Lists hypotheses currently flagged for human review — proxies bridge.py,
// which owns the Postgres hypothesis_reviews queue.

import { NextResponse } from 'next/server'

const BRIDGE_URL = process.env.BRIDGE_URL || 'http://localhost:8010'

export async function GET() {
  try {
    const res = await fetch(`${BRIDGE_URL}/reviews`, { cache: 'no-store' })

    if (!res.ok) {
      throw new Error(`bridge responded with ${res.status}`)
    }

    const reviews = await res.json()
    return NextResponse.json(reviews)
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: 502 })
  }
}
