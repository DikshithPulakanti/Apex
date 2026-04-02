'use client'

import { useEffect, useRef, useState } from 'react'
import cytoscape from 'cytoscape'

interface GraphData {
  nodes: any[]
  edges: any[]
}

const TYPE_COLORS: Record<string, string> = {
  hypothesis: '#facc15',  // yellow
  patent:     '#f59e0b',  // amber
  concept:    '#a78bfa',  // purple
  agent:      '#22d3ee',  // cyan
  paper:      '#60a5fa',  // blue
}

const STATUS_COLORS: Record<string, string> = {
  validated: '#4ade80',   // green
  rejected:  '#f87171',   // red
  proposed:  '#facc15',   // yellow
}

export default function KnowledgeGraph() {
  const containerRef = useRef<HTMLDivElement>(null)
  const cyRef = useRef<cytoscape.Core | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<any>(null)

  useEffect(() => {
    async function loadGraph() {
      try {
        const res = await fetch('/api/graph')
        if (!res.ok) throw new Error('Failed to load graph')
        const data: GraphData = await res.json()

        if (!containerRef.current) return

        // Filter out edges that reference missing nodes
        const nodeIds = new Set(data.nodes.map((n: any) => n.data.id))
        const validEdges = data.edges.filter(
          (e: any) => nodeIds.has(e.data.source) && nodeIds.has(e.data.target)
        )

        const cy = cytoscape({
          container: containerRef.current,
          elements: [...data.nodes, ...validEdges],
          style: [
            {
              selector: 'node',
              style: {
                'label': 'data(label)',
                'font-size': '8px',
                'color': '#e5e7eb',
                'text-wrap': 'ellipsis',
                'text-max-width': '80px',
                'text-valign': 'bottom',
                'text-margin-y': 4,
                'width': 20,
                'height': 20,
                'border-width': 1,
                'border-color': '#374151',
              }
            },
            {
              selector: 'node[type="hypothesis"]',
              style: {
                'background-color': '#facc15',
                'width': 30,
                'height': 30,
                'shape': 'diamond',
              }
            },
            {
              selector: 'node[type="hypothesis"][status="validated"]',
              style: { 'background-color': '#4ade80', 'border-color': '#16a34a', 'border-width': 2 }
            },
            {
              selector: 'node[type="hypothesis"][status="rejected"]',
              style: { 'background-color': '#f87171', 'border-color': '#dc2626', 'border-width': 2 }
            },
            {
              selector: 'node[type="patent"]',
              style: {
                'background-color': '#f59e0b',
                'width': 28,
                'height': 28,
                'shape': 'star',
              }
            },
            {
              selector: 'node[type="concept"]',
              style: {
                'background-color': '#a78bfa',
                'shape': 'ellipse',
              }
            },
            {
              selector: 'node[type="agent"]',
              style: {
                'background-color': '#22d3ee',
                'width': 25,
                'height': 25,
                'shape': 'hexagon',
              }
            },
            {
              selector: 'node[type="paper"]',
              style: {
                'background-color': '#60a5fa',
                'width': 14,
                'height': 14,
                'shape': 'round-rectangle',
                'font-size': '6px',
              }
            },
            {
              selector: 'edge',
              style: {
                'width': 1,
                'line-color': '#374151',
                'target-arrow-color': '#374151',
                'target-arrow-shape': 'triangle',
                'arrow-scale': 0.6,
                'curve-style': 'bezier',
                'opacity': 0.5,
              }
            },
            {
              selector: 'edge[type="DERIVED_FROM"]',
              style: { 'line-color': '#f59e0b', 'target-arrow-color': '#f59e0b', 'width': 2, 'opacity': 0.8 }
            },
            {
              selector: 'edge[type="HAS_CONCEPT"]',
              style: { 'line-color': '#6366f1', 'target-arrow-color': '#6366f1', 'opacity': 0.3 }
            },
          ],
          layout: {
            name: 'cose',
            animate: false,
            nodeRepulsion: () => 8000,
            idealEdgeLength: () => 80,
            gravity: 0.3,
            padding: 40,
          },
          minZoom: 0.3,
          maxZoom: 3,
        })

        cy.on('tap', 'node', (e) => {
          setSelected(e.target.data())
        })

        cy.on('tap', (e) => {
          if (e.target === cy) setSelected(null)
        })

        cyRef.current = cy
      } catch (e: any) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }

    loadGraph()

    return () => {
      cyRef.current?.destroy()
    }
  }, [])

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <h2 className="text-lg font-semibold text-gray-300">Knowledge Graph</h2>
        <div className="flex items-center gap-4 text-xs">
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded-full bg-blue-400 inline-block" /> Papers
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded-full bg-purple-400 inline-block" /> Concepts
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-green-400 inline-block" style={{ clipPath: 'polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%)' }} /> Hypotheses
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 bg-amber-400 inline-block" style={{ clipPath: 'polygon(50% 0%, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%)' }} /> Patents
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded-full bg-cyan-400 inline-block" /> Agents
          </span>
        </div>
      </div>

      {loading && (
        <div className="h-[500px] flex items-center justify-center text-gray-500">
          Loading graph...
        </div>
      )}
      {error && (
        <div className="h-[500px] flex items-center justify-center text-red-400">
          Error: {error}
        </div>
      )}

      <div ref={containerRef} className="h-[500px] w-full" />

      {selected && (
        <div className="px-4 py-3 border-t border-gray-800 text-sm">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-medium px-2 py-0.5 rounded-full"
              style={{ backgroundColor: TYPE_COLORS[selected.type] + '30', color: TYPE_COLORS[selected.type] }}>
              {selected.type}
            </span>
            {selected.status && (
              <span className="text-xs font-medium px-2 py-0.5 rounded-full"
                style={{ backgroundColor: STATUS_COLORS[selected.status] + '30', color: STATUS_COLORS[selected.status] }}>
                {selected.status}
              </span>
            )}
            <code className="text-xs text-gray-500">{selected.id}</code>
          </div>
          <p className="text-gray-300">{selected.label}</p>
          {selected.score != null && (
            <p className="text-xs text-gray-500 mt-1">Debate score: {selected.score}</p>
          )}
          {selected.connections != null && (
            <p className="text-xs text-gray-500 mt-1">Connections: {selected.connections}</p>
          )}
        </div>
      )}
    </div>
  )
}