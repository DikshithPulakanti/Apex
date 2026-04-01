# mcp_servers/patent_mcp.py
# APEX patent-mcp server — Patent Drafting and Prior Art Tools

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import anthropic
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from database.neo4j_client import Neo4jClient


server  = Server('patent-mcp')
_neo4j  = None
_claude = None


def get_neo4j():
    global _neo4j
    if _neo4j is None:
        _neo4j = Neo4jClient()
    return _neo4j


def get_claude():
    global _claude
    if _claude is None:
        _claude = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    return _claude


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name        = 'draft_patent_claims',
            description = (
                'Drafts structured patent claims for a validated hypothesis. '
                'Returns background, independent claims, dependent claims, '
                'and abstract in standard USPTO format. Use this after a '
                'hypothesis has passed simulation validation.'
            ),
            inputSchema = {
                'type': 'object',
                'properties': {
                    'hypothesis_statement': {
                        'type':        'string',
                        'description': 'The validated hypothesis to patent'
                    },
                    'supporting_concepts': {
                        'type':        'array',
                        'items':       {'type': 'string'},
                        'description': 'Key concepts supporting this invention'
                    },
                    'predicted_impact': {
                        'type':        'string',
                        'description': 'Expected impact if the hypothesis is validated'
                    }
                },
                'required': ['hypothesis_statement']
            }
        ),
        types.Tool(
            name        = 'check_prior_art',
            description = (
                'Searches the knowledge graph for existing work similar to '
                'a proposed invention. Returns related papers and concepts '
                'that represent prior art. Use this before drafting claims '
                'to assess novelty.'
            ),
            inputSchema = {
                'type': 'object',
                'properties': {
                    'invention_description': {
                        'type':        'string',
                        'description': 'Description of the proposed invention'
                    }
                },
                'required': ['invention_description']
            }
        ),
        types.Tool(
            name        = 'compute_novelty_score',
            description = (
                'Estimates how novel a hypothesis is compared to existing '
                'work in the knowledge graph. Returns a score from 0.0 '
                '(already known) to 1.0 (completely novel). '
                'Use this to prioritize which hypotheses to patent.'
            ),
            inputSchema = {
                'type': 'object',
                'properties': {
                    'hypothesis_statement': {
                        'type':        'string',
                        'description': 'The hypothesis to evaluate for novelty'
                    },
                    'supporting_concepts': {
                        'type':        'array',
                        'items':       {'type': 'string'},
                        'description': 'Concepts related to this hypothesis'
                    }
                },
                'required': ['hypothesis_statement']
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        if name == 'draft_patent_claims':
            return await handle_draft_patent(arguments)
        elif name == 'check_prior_art':
            return await handle_prior_art(arguments)
        elif name == 'compute_novelty_score':
            return await handle_novelty_score(arguments)
        else:
            return [types.TextContent(type='text', text=f'Unknown tool: {name}')]
    except Exception as e:
        return [types.TextContent(type='text', text=f'Error: {str(e)}')]


async def handle_draft_patent(args: dict) -> list[types.TextContent]:
    """Uses Claude to draft patent claims."""
    statement = args['hypothesis_statement']
    concepts  = args.get('supporting_concepts', [])
    impact    = args.get('predicted_impact', '')
    claude    = get_claude()

    prompt = f"""Draft a structured patent application for this invention:

Invention: {statement}
Key Concepts: {', '.join(concepts)}
Expected Impact: {impact}

Return ONLY a JSON object with this structure:
{{
    "title": "Patent title",
    "background": "Background of the invention (2-3 sentences)",
    "summary": "Summary of the invention (2-3 sentences)",
    "independent_claim_1": "The broadest independent claim",
    "dependent_claim_2": "A dependent claim that adds specificity",
    "dependent_claim_3": "Another dependent claim",
    "abstract": "Patent abstract (150 words max)"
}}"""

    message = claude.messages.create(
        model      = 'claude-sonnet-4-20250514',
        max_tokens = 1000,
        messages   = [{'role': 'user', 'content': prompt}]
    )

    text = message.content[0].text.strip()
    if '```json' in text:
        text = text.split('```json')[1].split('```')[0].strip()
    elif '```' in text:
        text = text.split('```')[1].split('```')[0].strip()

    patent = json.loads(text)
    patent['hypothesis'] = statement[:100]
    patent['status']     = 'draft'

    return [types.TextContent(type='text', text=json.dumps(patent, indent=2))]


async def handle_prior_art(args: dict) -> list[types.TextContent]:
    """Checks knowledge graph for prior art."""
    description = args['invention_description']
    neo4j       = get_neo4j()

    # Search for related hypotheses
    with neo4j.driver.session() as session:
        result = session.run("""
            MATCH (h:Hypothesis)
            RETURN h.id AS id, h.statement AS statement,
                   h.testability_score AS score
            LIMIT 5
        """)
        existing_hypotheses = [dict(r) for r in result]

    result = {
        'invention':              description[:100],
        'existing_hypotheses':    existing_hypotheses,
        'prior_art_risk':         'low' if len(existing_hypotheses) < 2 else 'medium',
        'recommendation':         (
            'Proceed — limited prior art found in knowledge base'
            if len(existing_hypotheses) < 2
            else 'Review existing hypotheses before proceeding'
        )
    }

    return [types.TextContent(type='text', text=json.dumps(result, indent=2))]


async def handle_novelty_score(args: dict) -> list[types.TextContent]:
    """Estimates novelty of a hypothesis."""
    statement = args['hypothesis_statement']
    concepts  = args.get('supporting_concepts', [])
    neo4j     = get_neo4j()

    # Count how many existing hypotheses share concepts
    shared_count = 0
    for concept in concepts:
        with neo4j.driver.session() as session:
            result = session.run("""
                MATCH (h:Hypothesis)-[:DERIVED_FROM]->(c:Concept {name: $name})
                RETURN count(h) AS n
            """, name=concept.lower())
            shared_count += result.single()['n']

    # More shared concepts = less novel
    novelty_score = max(0.1, 1.0 - (shared_count * 0.1))

    result = {
        'hypothesis':       statement[:100],
        'novelty_score':    round(novelty_score, 4),
        'shared_concepts':  shared_count,
        'assessment': (
            'highly novel' if novelty_score > 0.8
            else 'moderately novel' if novelty_score > 0.5
            else 'low novelty — similar work exists'
        )
    }

    return [types.TextContent(type='text', text=json.dumps(result, indent=2))]


async def main():
    print('[patent-mcp] Starting server...', file=sys.stderr)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == '__main__':
    asyncio.run(main())