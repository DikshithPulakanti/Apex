# tests/test_harvester.py
# Unit tests for the Harvester agent's node logic — DB/embedder mocked.

from agents import harvester


def test_search_papers_returns_hybrid_results(fake_weaviate, fake_embedder):
    fake_weaviate.hybrid_search.return_value = [
        {'paper_id': '2301.00001', 'title': 'GNNs for Drug Discovery', 'abstract': '...'},
    ]
    resources = {'weaviate': fake_weaviate, 'embedder': fake_embedder}
    state = {'query': 'graph neural networks for drug discovery', 'papers_found': [],
             'concepts_extracted': [], 'papers_processed': 0, 'status': 'starting', 'error': ''}

    result = harvester.search_papers(state, resources)

    assert len(result['papers_found']) == 1
    fake_weaviate.hybrid_search.assert_called_once()
    _, kwargs = fake_weaviate.hybrid_search.call_args
    assert kwargs['alpha'] == 0.7


def test_search_papers_handles_errors_gracefully(fake_weaviate, fake_embedder):
    fake_weaviate.hybrid_search.side_effect = RuntimeError('weaviate down')
    resources = {'weaviate': fake_weaviate, 'embedder': fake_embedder}
    state = {'query': 'x', 'papers_found': [], 'concepts_extracted': [],
             'papers_processed': 0, 'status': 'starting', 'error': ''}

    result = harvester.search_papers(state, resources)

    assert result['papers_found'] == []
    assert 'weaviate down' in result['error']
    assert result['status'] == 'search failed'


def test_extract_concepts_skips_when_no_papers(fake_neo4j):
    resources = {'neo4j': fake_neo4j, 'extractor': None}
    state = {'query': 'x', 'papers_found': [], 'concepts_extracted': [],
             'papers_processed': 0, 'status': '', 'error': ''}

    result = harvester.extract_concepts(state, resources)

    assert result['concepts_extracted'] == []
    assert result['status'] == 'no papers to extract from'


def test_extract_concepts_deduplicates_and_links(fake_neo4j):
    class FakeExtractor:
        def extract_concepts(self, text, max_concepts=6):
            return ['graph neural networks', 'drug discovery']

    resources = {'neo4j': fake_neo4j, 'extractor': FakeExtractor()}
    state = {
        'query': 'x',
        'papers_found': [
            {'paper_id': 'p1', 'abstract': 'about graph neural networks and drug discovery'},
            {'paper_id': 'p2', 'abstract': 'about graph neural networks'},
        ],
        'concepts_extracted': [], 'papers_processed': 0, 'status': '', 'error': '',
    }

    result = harvester.extract_concepts(state, resources)

    assert set(result['concepts_extracted']) == {'graph neural networks', 'drug discovery'}
    assert fake_neo4j.upsert_concept.call_count == 4  # 2 concepts x 2 papers
    assert fake_neo4j.link_paper_to_concept.call_count == 4


def test_insert_to_graph_rebuilds_cooccurrence(fake_neo4j):
    resources = {'neo4j': fake_neo4j}
    state = {'query': 'x', 'papers_found': [{'paper_id': 'p1'}], 'concepts_extracted': [],
             'papers_processed': 0, 'status': '', 'error': ''}

    result = harvester.insert_to_graph(state, resources)

    fake_neo4j.build_concept_cooccurrence.assert_called_once()
    assert result['papers_processed'] == 1
