def retrieve_with_fallback(collection, query: str, threshold: float = 0.5):
    """
    Queries ChromaDB and calculates similarity.
    For cosine distance: similarity = 1.0 - distance.
    """
    results = collection.query(
        query_texts=[query],
        n_results=1,
        include=["documents", "distances", "metadatas"],
    )

    if not results or not results["documents"] or not results["documents"][0]:
        return None, 0.0, "No documents indexed."

    distance = results["distances"][0][0]
    similarity = 1.0 - distance
    matched_doc = results["documents"][0][0]

    if similarity < threshold:
        return (
            None,
            similarity,
            f"Confidence below threshold ({similarity:.3f} < {threshold}). Redirecting to support.",
        )

    return matched_doc, similarity, None
