"""
Central repository for mock data, fallback text, and sample prompts.
"""

# Corpus used to simulate LLM token generation
MOCK_LLM_CORPUS = (
    "FastAPI provides native asynchronous support through Starlette. "
    "When handling Server-Sent Events (SSE), connection monitoring is critical. "
    "If a user closes their browser tab or clicks cancel, standard generators might "
    "continue producing tokens in the background, consuming valuable compute. "
    "By polling `request.is_disconnected()`, we can terminate upstream generators immediately "
    "and release precious GPU and memory resources."
)

# Default request parameters for testing
DEFAULT_TEST_PAYLOAD = {
    "prompt": "Explain why connection tracking matters in LLM inference.",
    "max_tokens": 50,
    "delay_per_token": 0.1,
}

# Sample prompt choices for the client UI or CLI tests
SAMPLE_PROMPTS = [
    "Explain why connection tracking matters in LLM inference.",
    "How does asyncio cooperatively multitask in Python?",
    "Compare Server-Sent Events (SSE) with WebSockets for LLMs.",
]
