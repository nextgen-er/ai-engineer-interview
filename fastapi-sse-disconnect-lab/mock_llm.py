import asyncio
from typing import AsyncGenerator
from mock_data import MOCK_LLM_CORPUS


async def mock_llm_stream(
    prompt: str,
    max_tokens: int = 50,
    delay: float = 0.1,
) -> AsyncGenerator[str, None]:
    """
    Simulates an asynchronous token generation engine or an upstream API call.
    Yields one word/token at a time from the mock corpus.
    """
    prefix = f"Thinking about '{prompt}'... "
    full_text = prefix + MOCK_LLM_CORPUS
    words = full_text.split()

    for idx in range(max_tokens):
        word = words[idx % len(words)] + " "
        await asyncio.sleep(delay)
        yield word
