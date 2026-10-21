import tiktoken


class TokenAwareChunker:
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def chunk_text(
        self, text: str, chunk_size_tokens: int = 60, chunk_overlap_tokens: int = 15
    ) -> list[str]:
        """
        Recursively splits text on natural boundaries (newlines, sentences, spaces)
        ensuring each chunk fits strictly within subword token limits.
        """
        if chunk_size_tokens <= chunk_overlap_tokens:
            raise ValueError(
                "chunk_size_tokens must be greater than chunk_overlap_tokens"
            )

        # Split into initial paragraphs/sentences
        raw_paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        segments = []
        for p in raw_paragraphs:
            # Further split sentences if needed
            sentences = [s.strip() + "." for s in p.split(".") if s.strip()]
            segments.extend(sentences)

        chunks = []
        current_tokens: list[int] = []

        for segment in segments:
            segment_tokens = self.tokenizer.encode(" " + segment, disallowed_special=())

            # If a single sentence exceeds the chunk limit, split by tokens directly
            if len(segment_tokens) > chunk_size_tokens:
                start = 0
                step = chunk_size_tokens - chunk_overlap_tokens
                while start < len(segment_tokens):
                    sub_chunk = segment_tokens[start : start + chunk_size_tokens]
                    chunks.append(self.tokenizer.decode(sub_chunk).strip())
                    start += step
                continue

            if len(current_tokens) + len(segment_tokens) <= chunk_size_tokens:
                current_tokens.extend(segment_tokens)
            else:
                if current_tokens:
                    chunks.append(self.tokenizer.decode(current_tokens).strip())
                    # Retain overlap from end of previous chunk
                    current_tokens = (
                        current_tokens[-chunk_overlap_tokens:] + segment_tokens
                    )
                else:
                    current_tokens = segment_tokens

        if current_tokens:
            chunks.append(self.tokenizer.decode(current_tokens).strip())

        return chunks
