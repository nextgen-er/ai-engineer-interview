def chunk_text(text: str, chunk_size: int = 30, overlap: int = 10) -> list[str]:
    """
    Splits text into word chunks with a specified overlap window,
    ensuring whitespace is preserved between words.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be strictly greater than overlap.")

    # .split() splits on all whitespace (spaces, tabs, newlines)
    words = text.split()
    if not words:
        return []

    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        # Explicitly join words with a single space
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break

    return chunks
