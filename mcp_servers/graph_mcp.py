# mcp_servers/graph_mcp.py
# APEX graph-mcp server — Neo4j Knowledge Graph Tools

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import uuid
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from database.neo4j_client import Neo4jClient


# ── Initialize server ─────────────────────────────────────────────────────
server  = Server('graph-mcp')
_neo4j  = None


def get_neo4j():
    global _neo4j
    if _neo4j is None:
        _neo4j = Neo4jClient()
    return _neo4j


# ── Tool definitions ──────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name        = 'find_research_gaps',
            description = (
                'Find potential research gaps in the knowledge graph. '
                'Returns concept pairs that are both important (high PageRank) '
                'but rarely connected (low co-occurrence). These are unexplored '
                'cross-domain research opportunities. Use this to identify '
                'where novel hypotheses could be generated.'
            ),
            inputSchema = {
                'type': 'object',
                'properties': {
                    'min_pagerank': {
                        'type':        'number',
                        'description': 'Minimum PageRank score for concepts (default 0.3)',
                        'default':     0.3
                    },
                    'limit': {
                        'type':        'integer',
                        'description': 'Number of gaps to return (default 10)',
                        'default':     10
                    }
                },
                'required': []
            }
        ),
        types.Tool(
            name        = 'create_hypothesis',
            description = (
                'Store a new hypothesis node in the Neo4j knowledge graph. '
                'Links the hypothesis to its supporting concepts. '
                'Use this after generating a hypothesis to persist it.'
            ),
            inputSchema = {
                'type': 'object',
                'properties': {
                    'statement': {
                        'type':        'string',
                        'description': 'The hypothesis statement'
                    },
                    'rationale': {
                        'type':        'string',
                        'description': 'Why this hypothesis is plausible'
                    },
                    'supporting_concepts': {
                        'type':        'array',
                        'items':       {'type': 'string'},
                        'description': 'List of concept names that support this hypothesis'
                    },
                    'testability_score': {
                        'type':        'number',
                        'description': 'How testable is this? 0.0 to 1.0'
                    },
                    'predicted_impact': {
                        'type':        'string',
                        'description': 'What happens if this hypothesis is validated?'
                    }
                },
                'required': ['statement', 'rationale', 'testability_score']
            }
        ),
        types.Tool(
            name        = 'get_graph_stats',
            description = (
                'Returns current statistics of the APEX knowledge graph: '
                'number of papers, authors, concepts, hypotheses, and relationships. '
                'Use this to understand the current state of the knowledge base.'
            ),
            inputSchema = {
                'type':       'object',
                'properties': {},
                'required':   []
            }
        ),
        types.Tool(
            name        = 'get_top_concepts',
            description = (
                'Returns the most influential concepts in the knowledge graph '
                'ranked by PageRank score. High PageRank means this concept '
                'connects to many other important concepts. '
                'Use this to find central research topics.'
            ),
            inputSchema = {
                'type': 'object',
                'properties': {
                    'limit': {
                        'type':        'integer',
                        'description': 'Number of concepts to return (default 10)',
                        'default':     10
                    }
                },
                'required': []
            }
        ),
        types.Tool(
            name        = 'get_hypotheses',
            description = (
                'Returns all hypotheses stored in the knowledge graph. '
                'Use this to see what hypotheses have already been generated '
                'and avoid generating duplicates.'
            ),
            inputSchema = {
                'type': 'object',
                'properties': {
                    'limit': {
                        'type':        'integer',
                        'description': 'Number of hypotheses to return (default 10)',
                        'default':     10
                    }
                },
                'required': []
            }
        ),
    ]


# ── Tool execution ────────────────────────────────────────────────────────

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        if name == 'find_research_gaps':
            return await handle_find_gaps(arguments)
        elif name == 'create_hypothesis':
            return await handle_create_hypothesis(arguments)
        elif name == 'get_graph_stats':
            return await handle_get_stats(arguments)
        elif name == 'get_top_concepts':
            return await handle_get_top_concepts(arguments)
        elif name == 'get_hypotheses':
            return await handle_get_hypotheses(arguments)
        else:
            return [types.TextContent(type='text', text=f'Unknown tool: {name}')]
    except Exception as e:
        return [types.TextContent(type='text', text=f'Error: {str(e)}')]


async def handle_find_gaps(args: dict) -> list[types.TextContent]:
    neo4j       = get_neo4j()
    min_pagerank = args.get('min_pagerank', 0.3)
    limit        = args.get('limit', 10)
    gaps         = neo4j.find_research_gaps(min_pagerank=min_pagerank, limit=limit)
    result       = {'count': len(gaps), 'gaps': gaps}
    return [types.TextContent(type='text', text=json.dumps(result, indent=2))]


async def handle_create_hypothesis(args: dict) -> list[types.TextContent]:
    neo4j         = get_neo4j()
    hypothesis_id = f'hyp_{uuid.uuid4().hex[:12]}'

    query = """
        MERGE (h:Hypothesis {id: $id})
        SET h.statement         = $statement,
            h.rationale         = $rationale,
            h.testability_score = $testability_score,
            h.predicted_impact  = $predicted_impact,
            h.status            = 'proposed',
            h.created_by        = 'graph-mcp'
        RETURN h
    """
    with neo4j.driver.session() as session:
        session.run(query,
            id                = hypothesis_id,
            statement         = args['statement'],
            rationale         = args['rationale'],
            testability_score = args.get('testability_score', 0.5),
            predicted_impact  = args.get('predicted_impact', '')
        )

    # Link to concepts
    for concept in args.get('supporting_concepts', []):
        link_query = """
            MATCH (h:Hypothesis {id: $hyp_id})
            MATCH (c:Concept {name: $name})
            MERGE (h)-[:DERIVED_FROM]->(c)
        """
        with neo4j.driver.session() as session:
            session.run(link_query, hyp_id=hypothesis_id, name=concept.lower())

    result = {
        'hypothesis_id': hypothesis_id,
        'status':        'created',
        'statement':     args['statement']
    }
    return [types.TextContent(type='text', text=json.dumps(result, indent=2))]


async def handle_get_stats(args: dict) -> list[types.TextContent]:
    neo4j = get_neo4j()
    stats = neo4j.get_stats()

    # Add hypothesis count
    with neo4j.driver.session() as session:
        result = session.run('MATCH (h:Hypothesis) RETURN count(h) AS n')
        stats['hypotheses'] = result.single()['n']

    return [types.TextContent(type='text', text=json.dumps(stats, indent=2))]


async def handle_get_top_concepts(args: dict) -> list[types.TextContent]:
    neo4j = get_neo4j()
    limit = args.get('limit', 10)

    with neo4j.driver.session() as session:
        result = session.run("""
            MATCH (c:Concept)
            WHERE c.pagerank IS NOT NULL
            RETURN c.name AS name, c.pagerank AS pagerank,
                   c.community AS community, c.betweenness AS betweenness
            ORDER BY c.pagerank DESC
            LIMIT $limit
        """, limit=limit)
        concepts = [dict(r) for r in result]

    return [types.TextContent(type='text', text=json.dumps({'concepts': concepts}, indent=2))]


async def handle_get_hypotheses(args: dict) -> list[types.TextContent]:
    neo4j = get_neo4j()
    limit = args.get('limit', 10)

    with neo4j.driver.session() as session:
        result = session.run("""
            MATCH (h:Hypothesis)
            RETURN h.id AS id, h.statement AS statement,
                   h.testability_score AS testability_score,
                   h.status AS status, h.created_by AS created_by
            ORDER BY h.testability_score DESC
            LIMIT $limit
        """, limit=limit)
        hypotheses = [dict(r) for r in result]

    return [types.TextContent(type='text', text=json.dumps({'hypotheses': hypotheses}, indent=2))]


# ── Run server ────────────────────────────────────────────────────────────

async def main():
    print('[graph-mcp] Starting server...', file=sys.stderr)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == '__main__':
    asyncio.run(main())