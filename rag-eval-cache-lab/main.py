"""
Main driver script testing:
1. Semantic Cache Hit/Miss logic
2. Exact Token Cost Calculation
3. Context Precision & Faithfulness Metrics
"""

import numpy as np
from rag_engine.cache import SemanticCache
from rag_engine.token_tracker import TokenCostTracker, ModelCostProfile
from rag_engine.metrics import RAGEvaluator

# Deterministic Embedding helper (Uses SentenceTransformer if installed, otherwise simple TF-IDF hash)
try:
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def get_embedding(text: str) -> np.ndarray:
        return encoder.encode(text)
except ImportError:
    print(
        "[WARN] sentence-transformers not found. Using local deterministic hash embedding fallback."
    )

    def get_embedding(text: str, dim: int = 128) -> np.ndarray:
        vec = np.zeros(dim)
        for i, word in enumerate(text.lower().split()):
            idx = abs(hash(word)) % dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / (norm if norm > 0 else 1.0)


def mock_rag_generate(query: str, retrieved_context: list[str]) -> str:
    """Simulated generation based on retrieval."""
    return f"Based on knowledge, {retrieved_context[0]}"


def main():
    print("================================================================")
    print(" RAG EVALUATION & SEMANTIC CACHING PIPELINE (1.5H LAB)")
    print("================================================================\n")

    # 1. Initialize Tracker & Cache
    # Example rates: GPT-4o-mini ($0.15 / 1M prompt, $0.60 / 1M completion)
    cost_profile = ModelCostProfile(
        input_cost_per_million=0.15, output_cost_per_million=0.60
    )
    tracker = TokenCostTracker(cost_profile)
    cache = SemanticCache(similarity_threshold=0.82)

    # 2. Knowledge Base Contexts
    retrieved_db = {
        "q1": [
            "Context 1: Vector embeddings map high-dimensional words into continuous dense vectors.",
            "Context 2: Cosine distance measures vector angular difference independently of magnitude.",
            "Context 3: Python memory allocation is managed by the private heap space.",  # Irrelevant noise
        ]
    }

    # Test Case 1: First Query (Cold - Cache Miss)
    q1 = "What do vector embeddings do?"
    q1_emb = get_embedding(q1)

    print(f"[*] Query 1: '{q1}'")
    cached_resp, sim_score = cache.get(q1, q1_emb)

    if cached_resp is None:
        print(f" -> Cache MISS (Similarity: {sim_score:.3f}). Invoking LLM...")
        gen_answer = (
            "Vector embeddings map high-dimensional words into continuous dense vectors. "
            "Cosine distance measures vector angular difference independently of magnitude."
        )
        cache.put(q1, q1_emb, gen_answer)
        tx = tracker.record_transaction(
            prompt=q1 + " " + " ".join(retrieved_db["q1"]), completion=gen_answer
        )
        print(f" -> Generated & Cached. Cost: ${tx['cost_usd']:.8f}")

    # Test Case 2: Semantically Equivalent Query (Cache Hit)
    q2 = "Explain what vector embeddings are doing"
    q2_emb = get_embedding(q2)

    print(f"\n[*] Query 2: '{q2}' (Paraphrased)")
    cached_resp, sim_score = cache.get(q2, q2_emb)

    if cached_resp:
        print(f" -> Cache HIT! (Cosine Similarity: {sim_score:.4f})")
        print(f" -> Returned Cached Response: '{cached_resp}'")
        tx = tracker.record_transaction(
            prompt=q2, completion=cached_resp, is_cache_hit=True
        )
        print(f" -> Saved API Call! Cost: ${tx['cost_usd']:.8f}")

    # 3. Context Precision & Faithfulness Evaluation
    print("\n----------------------------------------------------------------")
    print(" METRICS EVALUATION")
    print("----------------------------------------------------------------")
    ground_truth = (
        "Vector embeddings map high-dimensional words into continuous dense vectors."
    )

    # Context Precision:
    contexts_case_a = retrieved_db[
        "q1"
    ]  # Contains 2 relevant chunks, then 1 irrelevant
    precision_a = RAGEvaluator.context_precision(contexts_case_a, ground_truth)
    print(f"Scenario A (Good Ranking): Context Precision = {precision_a:.4f}")

    # Swapping irrelevant chunk to index 0 (Retrieval degradation)
    contexts_case_b = [contexts_case_a[2], contexts_case_a[0], contexts_case_a[1]]
    precision_b = RAGEvaluator.context_precision(contexts_case_b, ground_truth)
    print(f"Scenario B (Top Rank Irrelevant): Context Precision = {precision_b:.4f}")

    # Faithfulness Check:
    faithful_answer = (
        "Vector embeddings map high-dimensional words into continuous dense vectors."
    )
    unfaithful_answer = (
        "Vector embeddings map high-dimensional words into continuous dense vectors. "
        "They were invented in 1642 by Blaise Pascal to measure planetary orbits."  # Hallucinated statement
    )

    faith_score_1 = RAGEvaluator.faithfulness(faithful_answer, contexts_case_a)
    faith_score_2 = RAGEvaluator.faithfulness(unfaithful_answer, contexts_case_a)

    print(f"\nFaithfulness (Accurate Generation): {faith_score_1:.2f} (100% grounded)")
    print(f"Faithfulness (Hallucinated Sentence): {faith_score_2:.2f} (50% grounded)")

    # 4. Observability Summary
    print("\n----------------------------------------------------------------")
    print(" TOKEN & COST OBSERVABILITY SUMMARY")
    print("----------------------------------------------------------------")
    summary = tracker.summary()
    for k, v in summary.items():
        print(f" - {k}: {v}")


if __name__ == "__main__":
    main()
