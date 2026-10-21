import logging
import math
from typing import Optional, Tuple
from sentence_transformers import CrossEncoder
from config import settings

logger = logging.getLogger(__name__)


class GuardedRetriever:
    def __init__(self, collection):
        self.collection = collection
        logger.info(f"Loading Cross-Encoder model: {settings.RERANKER_MODEL_NAME}")
        self.reranker = CrossEncoder(settings.RERANKER_MODEL_NAME)

    def retrieve(
        self, query: str, threshold: Optional[float] = None
    ) -> Tuple[Optional[str], float, Optional[str]]:
        cutoff = threshold if threshold is not None else settings.CONFIDENCE_THRESHOLD

        # 1. Fetch candidates from ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=settings.VECTOR_TOP_K,
            include=["documents", "distances", "metadatas"],
        )

        candidates = results.get("documents", [[]])[0]
        if not candidates:
            return None, 0.0, "Vector index returned no candidate matches."

        # 2. Score candidates with Cross-Encoder
        query_document_pairs = [[query, doc] for doc in candidates]
        raw_scores = self.reranker.predict(query_document_pairs)

        # Handle single float or numpy array
        if not hasattr(raw_scores, "__len__"):
            raw_scores = [raw_scores]

        # Convert logits to calibrated sigmoid probabilities [0, 1]
        # Temperature scaling (T=2.0) smooths extreme logit bounds
        temperature = 2.0
        calibrated_scores = [
            1.0 / (1.0 + math.exp(-float(score) / temperature)) for score in raw_scores
        ]

        best_idx = max(
            range(len(calibrated_scores)), key=lambda i: calibrated_scores[i]
        )
        best_score = calibrated_scores[best_idx]
        best_document = candidates[best_idx]

        logger.info(
            f"Query: '{query}' | Candidates: {len(candidates)} | "
            f"Raw Logit: {raw_scores[best_idx]:.3f} | Calibrated Score: {best_score:.4f}"
        )

        # 3. Guardrail Threshold
        if best_score < cutoff:
            reason = (
                f"Confidence below threshold ({best_score:.3f} < {cutoff:.3f}). "
                "Redirecting to formal support."
            )
            return None, best_score, reason

        return best_document, best_score, None
