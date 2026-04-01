# agents/orchestrator.py
# APEX Orchestrator — runs Harvester then Reasoner in sequence

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from agents.harvester import build_harvester, get_resources as harvester_resources
from agents.reasoner  import build_reasoner, get_resources as reasoner_resources, empty_hypothesis


def run_apex_pipeline(topic: str) -> dict:
    """
    Runs the full APEX pipeline for a given research topic.

    Step 1: Harvester finds relevant papers and extracts concepts
    Step 2: Reasoner finds research gaps and generates a hypothesis

    PARAMETERS:
        topic : research topic to explore, e.g. 'protein folding with graph neural networks'

    RETURNS:
        dict with pipeline results
    """
    print(f'\n{"="*60}')
    print(f'APEX PIPELINE — Topic: {topic}')
    print(f'{"="*60}\n')

    results = {
        'topic':          topic,
        'papers_found':   0,
        'concepts_found': 0,
        'hypothesis':     None,
        'hypothesis_id':  '',
        'status':         'starting'
    }

    # ── Phase 1: Harvester ────────────────────────────────────────────────
    print('PHASE 1: Harvester Agent')
    print('-' * 40)

    h_resources = harvester_resources()
    harvester   = build_harvester(h_resources)

    h_state = harvester.invoke({
        'query':              topic,
        'papers_found':       [],
        'concepts_extracted': [],
        'papers_processed':   0,
        'status':             'starting',
        'error':              ''
    })

    results['papers_found']   = len(h_state['papers_found'])
    results['concepts_found'] = len(h_state['concepts_extracted'])

    print(f'\nHarvester complete:')
    print(f'  Papers found:   {results["papers_found"]}')
    print(f'  Concepts found: {results["concepts_found"]}')

    # Close harvester resources
    h_resources['neo4j'].close()
    h_resources['weaviate'].close()

    # ── Phase 2: Reasoner ─────────────────────────────────────────────────
    print('\nPHASE 2: Reasoner Agent')
    print('-' * 40)

    r_resources = reasoner_resources()
    reasoner    = build_reasoner(r_resources)

    r_state = reasoner.invoke({
        'seed_concept':   topic,
        'gaps_found':     [],
        'context_papers': [],
        'hypothesis':     empty_hypothesis(),
        'hypothesis_id':  '',
        'attempts':       0,
        'status':         'starting',
        'error':          ''
    })

    results['hypothesis']    = r_state['hypothesis']
    results['hypothesis_id'] = r_state['hypothesis_id']
    results['status']        = 'complete'

    # Close reasoner resources
    r_resources['neo4j'].close()
    r_resources['weaviate'].close()

    # ── Summary ───────────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print('APEX PIPELINE COMPLETE')
    print(f'{"="*60}')
    print(f'Topic:          {topic}')
    print(f'Papers found:   {results["papers_found"]}')
    print(f'Concepts found: {results["concepts_found"]}')
    print(f'Hypothesis ID:  {results["hypothesis_id"]}')

    if results['hypothesis'] and results['hypothesis']['statement']:
        print(f'\nGenerated Hypothesis:')
        print(f'  {results["hypothesis"]["statement"]}')
        print(f'\nTestability: {results["hypothesis"]["testability_score"]}')
        print(f'Impact:      {results["hypothesis"]["predicted_impact"][:120]}')

    return results


if __name__ == '__main__':
    # Run on 3 different topics
    topics = [
        'graph neural networks for drug discovery',
        'reinforcement learning for scientific hypothesis generation',
        'large language models for materials science',
    ]

    all_results = []
    for topic in topics:
        result = run_apex_pipeline(topic)
        all_results.append(result)
        print('\n')

    print(f'\n{"="*60}')
    print(f'FINAL SUMMARY — {len(all_results)} topics processed')
    print(f'{"="*60}')
    for r in all_results:
        print(f'\nTopic: {r["topic"]}')
        print(f'  Papers: {r["papers_found"]} | Concepts: {r["concepts_found"]}')
        print(f'  Hypothesis ID: {r["hypothesis_id"]}')
        if r["hypothesis"]:
            print(f'  Statement: {r["hypothesis"]["statement"][:80]}...')