// app/api/events/route.ts
// Proxies bridge.py's /events/recent — a real Kafka consumer — instead of
// faking a timeline from Neo4j state. bridge.py owns the actual consumer;
// this route just forwards its response to the dashboard.

import { NextResponse } from 'next/server'

const BRIDGE_URL = process.env.BRIDGE_URL || 'http://localhost:8010'

export async function GET() {
  try {
    const res = await fetch(`${BRIDGE_URL}/events/recent?limit=50`, {
      cache: 'no-store',
    })

    if (!res.ok) {
      throw new Error(`bridge responded with ${res.status}`)
    }

    const events = await res.json()
    return NextResponse.json(events)
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: 502 })
  }
}
