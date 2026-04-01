# mcp_servers/paper_mcp.py
# APEX paper-mcp server — Scientific Literature Tools

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from database.weaviate_client import WeaviateClient
from database.embedder import Embedder
from database.neo4j_client import Neo4jClient


# ── Initialize server ─────────────────────────────────────────────────────
server = Server('paper-mcp')

# Resources created once at startup
_weaviate = None
_embedder = None
_neo4j    = None


def get_weaviate():
    global _weaviate
    if _weaviate is None:
        _weaviate = WeaviateClient()
    return _weaviate


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def get_neo4j():
    global _neo4j
    if _neo4j is None:
        _neo4j = Neo4jClient()
    return _neo4j


# ── Tool 1: search_papers ─────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Lists all tools this MCP server exposes."""
    return [
        types.Tool(
            name        = 'search_papers',
            description = (
                'Search for scientific papers semantically similar to a query. '
                'Uses hybrid vector + keyword search. Returns paper titles, '
                'abstracts, and relevance scores. Use this when you need to find '
                'papers related to a research topic or concept.'
            ),
            inputSchema = {
                'type': 'object',
                'properties': {
                    'query': {
                        'type':        'string',
                        'description': 'The search query — a research topic or question'
                    },
                    'limit': {
                        'type':        'integer',
                        'description': 'Number of papers to return (default 5, max 20)',
                        'default':     5
                    },
                    'alpha': {
                        'type':        'number',
                        'description': 'Balance between semantic (1.0) and keyword (0.0) search. Default 0.7',
                        'default':     0.7
                    }
                },
                'required': ['query']
            }
        ),
        types.Tool(
            name        = 'get_paper_details',
            description = (
                'Get full details of a specific paper by its ID, including '
                'title, abstract, year, authors, and concepts. Use this when '
                'you need deep information about a specific paper.'
            ),
            inputSchema = {
                'type': 'object',
                'properties': {
                    'paper_id': {
                        'type':        'string',
                        'description': 'The unique paper ID from arXiv, e.g. 2301.00001'
                    }
                },
                'required': ['paper_id']
            }
        ),
        types.Tool(
            name        = 'get_paper_concepts',
            description = (
                'Get the key concepts extracted from a paper. '
                'Returns a list of technical concept phrases. '
                'Use this to understand what a paper is about conceptually.'
            ),
            inputSchema = {
                'type': 'object',
                'properties': {
                    'paper_id': {
                        'type':        'string',
                        'description': 'The unique paper ID'
                    }
                },
                'required': ['paper_id']
            }
        ),
        types.Tool(
            name        = 'get_paper_neighbors',
            description = (
                'Find papers that share authors with a given paper. '
                'Returns related papers from the same research group. '
                'Use this to explore an author\'s body of work.'
            ),
            inputSchema = {
                'type': 'object',
                'properties': {
                    'paper_id': {
                        'type':        'string',
                        'description': 'The unique paper ID'
                    }
                },
                'required': ['paper_id']
            }
        ),
        types.Tool(
            name        = 'get_papers_by_year',
            description = (
                'Get all papers published in a specific year. '
                'Useful for understanding research trends over time.'
            ),
            inputSchema = {
                'type': 'object',
                'properties': {
                    'year': {
                        'type':        'integer',
                        'description': 'The publication year, e.g. 2023'
                    }
                },
                'required': ['year']
            }
        ),
    ]


# ── Tool execution ────────────────────────────────────────────────────────

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Routes tool calls to the correct handler."""

    try:
        if name == 'search_papers':
            return await handle_search_papers(arguments)
        elif name == 'get_paper_details':
            return await handle_get_paper_details(arguments)
        elif name == 'get_paper_concepts':
            return await handle_get_paper_concepts(arguments)
        elif name == 'get_paper_neighbors':
            return await handle_get_paper_neighbors(arguments)
        elif name == 'get_papers_by_year':
            return await handle_get_papers_by_year(arguments)
        else:
            return [types.TextContent(
                type = 'text',
                text = f'Unknown tool: {name}'
            )]
    except Exception as e:
        return [types.TextContent(
            type = 'text',
            text = f'Error executing {name}: {str(e)}'
        )]


async def handle_search_papers(args: dict) -> list[types.TextContent]:
    query   = args['query']
    limit   = args.get('limit', 5)
    alpha   = args.get('alpha', 0.7)

    embedder = get_embedder()
    weaviate = get_weaviate()

    query_vec = embedder.embed_text(query)
    papers    = weaviate.hybrid_search(query, query_vec, limit=limit, alpha=alpha)

    result = {
        'query':   query,
        'count':   len(papers),
        'papers':  [
            {
                'paper_id': p.get('paper_id', ''),
                'title':    p.get('title', ''),
                'abstract': p.get('abstract', '')[:300],
                'year':     p.get('year', 0),
                'score':    round(p.get('score', 0), 4)
            }
            for p in papers
        ]
    }

    return [types.TextContent(type='text', text=json.dumps(result, indent=2))]


async def handle_get_paper_details(args: dict) -> list[types.TextContent]:
    paper_id = args['paper_id']
    neo4j    = get_neo4j()
    paper    = neo4j.get_paper(paper_id)

    if not paper:
        return [types.TextContent(type='text', text=f'Paper not found: {paper_id}')]

    # Remove embedding from output — too large
    paper.pop('embedding', None)
    return [types.TextContent(type='text', text=json.dumps(paper, indent=2))]


async def handle_get_paper_concepts(args: dict) -> list[types.TextContent]:
    paper_id = args['paper_id']
    neo4j    = get_neo4j()
    concepts = neo4j.get_concepts_for_paper(paper_id)

    result = {'paper_id': paper_id, 'concepts': concepts}
    return [types.TextContent(type='text', text=json.dumps(result, indent=2))]


async def handle_get_paper_neighbors(args: dict) -> list[types.TextContent]:
    paper_id  = args['paper_id']
    neo4j     = get_neo4j()
    neighbors = neo4j.get_paper_neighbors(paper_id)

    # Remove embeddings from output
    for n in neighbors:
        n.pop('embedding', None)

    result = {'paper_id': paper_id, 'count': len(neighbors), 'neighbors': neighbors}
    return [types.TextContent(type='text', text=json.dumps(result, indent=2))]


async def handle_get_papers_by_year(args: dict) -> list[types.TextContent]:
    year   = args['year']
    neo4j  = get_neo4j()
    papers = neo4j.get_papers_by_year(year)

    # Remove embeddings
    for p in papers:
        p.pop('embedding', None)

    result = {'year': year, 'count': len(papers), 'papers': papers[:10]}
    return [types.TextContent(type='text', text=json.dumps(result, indent=2))]


# ── Run server ────────────────────────────────────────────────────────────

async def main():
    print('[paper-mcp] Starting server...', file=sys.stderr)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == '__main__':
    asyncio.run(main())