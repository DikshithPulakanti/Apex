# database/embedder.py
# APEX Embedding Engine — converts text to vectors

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Optional


class Embedder:
    """
    Converts text into numerical vectors (embeddings).
    
    Uses the all-MiniLM-L6-v2 model — fast, small, good quality.
    Output: 384-dimensional vector for any input text.
    
    First run downloads the model (~90MB). Cached locally after that.
    """

    MODEL_NAME = 'all-MiniLM-L6-v2'

    def __init__(self):
        print(f'[Embedder] Loading model: {self.MODEL_NAME}')
        print(f'[Embedder] First run downloads ~90MB. Please wait...')
        self.model = SentenceTransformer(self.MODEL_NAME)
        print(f'[Embedder] Model ready.')

    def embed_text(self, text: str) -> list:
        """
        Converts one piece of text into a 384-dimensional vector.
        
        PARAMETERS:
            text: any string — abstract, title, hypothesis statement
            
        RETURNS:
            list of 384 floats representing the semantic meaning
        """
        if not text or not text.strip():
            return [0.0] * 384

        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list) -> list:
        """
        Converts many texts into vectors at once.
        Much faster than calling embed_text() in a loop.
        
        PARAMETERS:
            texts: list of strings
            
        RETURNS:
            list of embeddings, one per input text
        """
        if not texts:
            return []

        # Filter out empty strings
        valid_texts  = [t if t and t.strip() else ' ' for t in texts]
        embeddings   = self.model.encode(valid_texts, convert_to_numpy=True)
        return embeddings.tolist()

    def cosine_similarity(self, vec1: list, vec2: list) -> float:
        """
        Measures how similar two vectors are.
        Returns a value between -1 and 1.
        1.0  = identical meaning
        0.0  = unrelated
        -1.0 = opposite meaning
        """
        a = np.array(vec1)
        b = np.array(vec2)

        # Avoid division by zero
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))


# ── Test it directly ───────────────────────────────────────────────────────
if __name__ == '__main__':
    embedder = Embedder()

    print('\n--- Testing single embedding ---')
    vec = embedder.embed_text('attention mechanism in transformers')
    print(f'  Vector dimensions: {len(vec)}')
    print(f'  First 5 values:    {[round(v, 4) for v in vec[:5]]}')
    assert len(vec) == 384, 'Should be 384 dimensions'
    print('  ✓ Single embedding correct dimensions')

    print('\n--- Testing cosine similarity ---')
    text1 = 'attention mechanism in transformer neural networks'
    text2 = 'self-attention and multi-head attention in deep learning'
    text3 = 'recipe for making chocolate cake at home'

    vec1 = embedder.embed_text(text1)
    vec2 = embedder.embed_text(text2)
    vec3 = embedder.embed_text(text3)

    sim_12 = embedder.cosine_similarity(vec1, vec2)
    sim_13 = embedder.cosine_similarity(vec1, vec3)

    print(f'  Similarity (AI vs AI):   {sim_12:.4f}')
    print(f'  Similarity (AI vs cake): {sim_13:.4f}')

    assert sim_12 > sim_13, 'Similar texts should score higher than unrelated'
    print('  ✓ Similar texts score higher than unrelated texts')

    print('\n--- Testing batch embedding ---')
    texts = [
        'graph neural networks for molecular property prediction',
        'large language models for scientific discovery',
        'diffusion models for protein structure prediction',
    ]
    vecs = embedder.embed_batch(texts)
    print(f'  Embedded {len(vecs)} texts')
    assert len(vecs) == 3
    assert len(vecs[0]) == 384
    print('  ✓ Batch embedding working correctly')

    print('\n✅ Embedder working correctly.')