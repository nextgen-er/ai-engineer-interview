import logging
from pathlib import Path
from config import settings
from chunking import TokenAwareChunker
from indexer import get_chroma_client, get_or_create_collection, batch_upsert_chunks
from retriever import GuardedRetriever
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("RAGPipeline")


def main():
    # 1. Initialize client & collection
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    # 2. Ingest domain documents
    policy_path = Path("data/policy.txt")
    if not policy_path.exists():
        logger.error(f"Required document not found at {policy_path.resolve()}")
        return

    raw_text = policy_path.read_text(encoding="utf-8")
    chunker = TokenAwareChunker()
    chunks = chunker.chunk_text(
        raw_text,
        chunk_size_tokens=settings.CHUNK_SIZE_TOKENS,
        chunk_overlap_tokens=settings.CHUNK_OVERLAP_TOKENS,
    )

    batch_upsert_chunks(
        collection=collection,
        chunks=chunks,
        source_name=policy_path.name,
        category="finance",
    )

    # 3. Initialize Guarded Retriever
    retriever = GuardedRetriever(collection=collection)

    # 4. Evaluation Queries
    test_queries = [
        "Do I need a receipt for a $30 Uber ride?",  # Direct in-domain
        "How do I configure Docker containers?",  # Completely out-of-domain
        "What is the daily allowance cap on meals?",  # Paraphrased in-domain
    ]

    print("\n" + "=" * 70)
    print("RUNNING RETRIEVAL EVALUATIONS")
    print("=" * 70)

    for query in test_queries:
        print(f"\nEvaluating: '{query}'")
        matched_doc, score, error = retriever.retrieve(query)

        if matched_doc:
            print(f"Status: [PASS] Confidence Score: {score:.3f}")
            print(f"Retrieved Passage: {matched_doc}")
        else:
            print(f"Status: [FALLBACK TRIGGERED] Score: {score:.3f}")
            print(f"Reason: {error}")


if __name__ == "__main__":
    main()
