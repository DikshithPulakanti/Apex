# scrapers/concept_extractor.py
# APEX Concept Extractor — extracts key concepts from paper abstracts

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spacy
from collections import Counter


class ConceptExtractor:
    """
    Extracts key technical concepts from paper abstracts using NLP.

    Uses spaCy's en_core_web_lg model to identify noun phrases
    that represent technical concepts — things like:
    'graph neural network', 'attention mechanism', 'protein folding'
    """

    def __init__(self):
        print('[ConceptExtractor] Loading spaCy model...')
        self.nlp = spacy.load('en_core_web_lg')
        print('[ConceptExtractor] Model ready.')

        self.stop_concepts = {
            # Generic paper language
            'paper', 'method', 'approach', 'result', 'model', 'system',
            'work', 'study', 'task', 'problem', 'data', 'dataset',
            'experiment', 'performance', 'training', 'learning', 'network',
            'neural network', 'deep learning', 'machine learning',
            'state of the art', 'sota', 'et al', 'fig', 'table',
            # Pronouns and demonstratives
            'this paper', 'this work', 'this approach', 'this method',
            'this model', 'this study', 'this task', 'this framework',
            'our method', 'our approach', 'our model', 'our work',
            'the model', 'the method', 'the approach', 'the system',
            'the task', 'the problem', 'the dataset', 'the network',
            # Vague nouns
            'the ability', 'the capability', 'the performance',
            'the development', 'the process', 'the use', 'the need',
            'the risks', 'the benefits', 'the challenge', 'the challenges',
            'the results', 'the findings', 'the fact', 'the goal',
            'a key', 'a novel', 'a new', 'a simple', 'a unified',
            'this paradigm', 'this direction', 'this setting',
            'existing methods', 'existing approaches', 'existing benchmarks',
            'recent advances', 'recent work', 'prior work',
            'various tasks', 'various methods', 'various applications',
        }

    def extract_concepts(self, text: str, max_concepts: int = 10) -> list:
        """
        Extracts technical concept phrases from a piece of text.

        PARAMETERS:
            text         : the abstract or title to extract from
            max_concepts : maximum number of concepts to return

        RETURNS:
            list of concept strings
        """
        if not text or not text.strip():
            return []

        doc = self.nlp(text[:5000])

        concepts = []
        for chunk in doc.noun_chunks:
            concept = chunk.text.lower().strip()

            # Filter 1: must be at least 2 words
            if len(concept.split()) < 2:
                continue

            # Filter 2: must be reasonable length
            if len(concept) < 5 or len(concept) > 60:
                continue

            # Filter 3: remove stop concepts
            if concept in self.stop_concepts:
                continue

            # Filter 4: remove concepts with numbers
            if any(char.isdigit() for char in concept):
                continue

            # Filter 5: remove concepts starting with common determiners
            if concept.startswith(('the ', 'a ', 'an ', 'this ', 'our ',
                                   'their ', 'these ', 'those ', 'its ')):
                continue

            concepts.append(concept)

        # Remove duplicates while preserving order
        seen = set()
        unique_concepts = []
        for c in concepts:
            if c not in seen:
                seen.add(c)
                unique_concepts.append(c)

        return unique_concepts[:max_concepts]

    def extract_batch(self, texts: list, max_concepts: int = 10) -> list:
        """
        Extracts concepts from many texts at once.
        Uses spaCy's pipe() for efficiency.

        PARAMETERS:
            texts        : list of abstract strings
            max_concepts : max concepts per text

        RETURNS:
            list of lists — one concept list per input text
        """
        if not texts:
            return []

        clean_texts = [t[:5000] if t else '' for t in texts]

        all_concepts = []
        for doc in self.nlp.pipe(clean_texts, batch_size=50):
            concepts = []
            for chunk in doc.noun_chunks:
                concept = chunk.text.lower().strip()

                # Filter 1: must be at least 2 words
                if len(concept.split()) < 2:
                    continue

                # Filter 2: must be reasonable length
                if len(concept) < 5 or len(concept) > 60:
                    continue

                # Filter 3: remove stop concepts
                if concept in self.stop_concepts:
                    continue

                # Filter 4: remove concepts with numbers
                if any(char.isdigit() for char in concept):
                    continue

                # Filter 5: remove concepts starting with common determiners
                if concept.startswith(('the ', 'a ', 'an ', 'this ', 'our ',
                                       'their ', 'these ', 'those ', 'its ')):
                    continue

                concepts.append(concept)

            # Deduplicate
            seen = set()
            unique = []
            for c in concepts:
                if c not in seen:
                    seen.add(c)
                    unique.append(c)

            all_concepts.append(unique[:max_concepts])

        return all_concepts


# ── Test it directly ───────────────────────────────────────────────────────
if __name__ == '__main__':
    extractor = ConceptExtractor()

    print('\n--- Testing single extraction ---')
    abstract = """
    We propose a novel graph neural network architecture for molecular
    property prediction in drug discovery. Our attention mechanism
    enables the model to focus on relevant chemical substructures.
    Experiments on benchmark datasets demonstrate state-of-the-art
    performance in protein-ligand binding affinity prediction.
    """
    concepts = extractor.extract_concepts(abstract)
    print(f'Extracted {len(concepts)} concepts:')
    for c in concepts:
        print(f'  → {c}')

    assert len(concepts) > 0
    print('  ✓ Single extraction working')

    print('\n--- Testing batch extraction ---')
    abstracts = [
        "We introduce BERT for natural language understanding using transformer architecture.",
        "AlphaFold predicts protein structure using deep learning and multiple sequence alignment.",
        "Reinforcement learning from human feedback improves large language model alignment.",
    ]
    batch_results = extractor.extract_batch(abstracts)
    print(f'Processed {len(batch_results)} abstracts:')
    for i, concepts in enumerate(batch_results):
        print(f'  Abstract {i+1}: {concepts[:3]}')

    assert len(batch_results) == 3
    print('  ✓ Batch extraction working')

    print('\n✅ ConceptExtractor working correctly.')