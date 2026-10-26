"""
Semantic Query Caching using Vector Embeddings and NumPy Cosine Similarity.
"""

from typing import Optional, Tuple, List
import numpy as np


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes cosine similarity between two 1D vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))


class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.query_vectors: List[np.ndarray] = []
        self.cached_queries: List[str] = []
        self.cached_responses: List[str] = []

    def get(
        self, query: str, query_embedding: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """
        Looks up nearest query vector. Returns (response, similarity_score) if >= threshold.
        """
        if not self.query_vectors:
            return None, 0.0

        # Vectorized similarity against all cached queries
        matrix = np.vstack(self.query_vectors)  # (N, D)
        query_norm = np.linalg.norm(query_embedding)
        matrix_norms = np.linalg.norm(matrix, axis=1)

        denominators = matrix_norms * query_norm
        denominators[denominators == 0] = 1e-10

        similarities = np.dot(matrix, query_embedding) / denominators
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= self.similarity_threshold:
            return self.cached_responses[best_idx], best_score

        return None, best_score

    def put(self, query: str, query_embedding: np.ndarray, response: str) -> None:
        self.cached_queries.append(query)
        self.query_vectors.append(query_embedding)
        self.cached_responses.append(response)
