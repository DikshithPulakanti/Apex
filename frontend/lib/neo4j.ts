// lib/neo4j.ts
// Neo4j driver for Next.js API routes

import neo4j, { Driver } from 'neo4j-driver'

let driver: Driver | null = null

export function getDriver(): Driver {
  if (!driver) {
    driver = neo4j.driver(
      process.env.NEO4J_URI || 'bolt://localhost:7687',
      neo4j.auth.basic(
        process.env.NEO4J_USER || 'neo4j',
        process.env.NEO4J_PASSWORD || 'apexpassword'
      )
    )
  }
  return driver
}

export async function runQuery(cypher: string, params: Record<string, any> = {}) {
  const session = getDriver().session()
  try {
    const result = await session.run(cypher, params)
    return result.records.map(r => r.toObject())
  } finally {
    await session.close()
  }
}