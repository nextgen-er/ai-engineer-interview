from pathlib import Path
from chunking import chunk_text
from indexer import get_collection, index_chunks
from retriever import retrieve_with_fallback


def main():
    collection = get_collection()

    file_path = Path("data/policy.txt")
    if file_path.exists():
        raw_text = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(raw_text, chunk_size=30, overlap=10)
        print(f"Generated {len(chunks)} chunks from {file_path.name}")

        index_chunks(collection, chunks, category="finance")
        print("Indexed documents successfully.\n" + "-" * 50)

    else:
        print(f"Error: {file_path} not found")
        return

    test_queries = [
        "Do I need a receipt for a $30 Uber ride?",
        "How do I configure Docker container?",
        "What is the cap on daily food expenses?",
    ]

    threshold = 0.3

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        doc, similarity, error_or_fallback = retrieve_with_fallback(
            collection, query, threshold=threshold
        )

        if doc:
            print(f"Status: Match found (Similarity: {similarity:.3f})")
            print(f"Document snippet: {doc}")

        else:
            print(f"Status: Fallback Triggered")
            print(f"Reason: {error_or_fallback}")


if __name__ == "__main__":
    main()
