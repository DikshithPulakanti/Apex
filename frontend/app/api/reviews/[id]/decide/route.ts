// app/api/reviews/[id]/decide/route.ts
// Approves or rejects a pending hypothesis review — proxies bridge.py,
// which updates Postgres + Neo4j and (on approval) triggers the Inventor.

import { NextResponse } from 'next/server'

const BRIDGE_URL = process.env.BRIDGE_URL || 'http://localhost:8010'

export async function POST(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params
    const body = await request.json()

    const res = await fetch(`${BRIDGE_URL}/reviews/${id}/decide`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })

    const data = await res.json()
    return NextResponse.json(data, { status: res.status })
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: 502 })
  }
}
