import asyncio
import json
import logging
from typing import AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from mock_data import DEFAULT_TEST_PAYLOAD
from mock_llm import mock_llm_stream

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("sse-service")

app = FastAPI(title="Async SSE Streaming with Disconnect Detection")


class ChatRequest(BaseModel):
    prompt: str = Field(default=DEFAULT_TEST_PAYLOAD["prompt"])
    max_tokens: int = Field(default=DEFAULT_TEST_PAYLOAD["max_tokens"])
    delay_per_token: float = Field(default=DEFAULT_TEST_PAYLOAD["delay_per_token"])


async def event_generator(
    request: Request, body: ChatRequest
) -> AsyncGenerator[str, None]:
    session_id = id(request)
    logger.info(
        f"[{session_id}] Started generation for prompt: '{body.prompt[:30]}...'"
    )

    tokens_generated = 0
    try:
        async for token in mock_llm_stream(
            body.prompt, body.max_tokens, body.delay_per_token
        ):
            # Proactively check if client dropped connection before next yield
            if await request.is_disconnected():
                logger.warning(
                    f"[{session_id}] Client disconnected! Aborting generator. Tokens produced: {tokens_generated}"
                )
                break

            payload = json.dumps({"token": token, "index": tokens_generated})
            yield f"data: {payload}\n\n"
            tokens_generated += 1

        if not await request.is_disconnected():
            yield "data: [DONE]\n\n"
            logger.info(
                f"[{session_id}] Stream completed successfully ({tokens_generated} tokens)."
            )

    except asyncio.CancelledError:
        logger.warning(f"[{session_id}] Stream task cancelled via CancelledError.")
        raise
    finally:
        # Guarantee teardown logic runs regardless of how the stream ended
        logger.info(f"[{session_id}] Teardown completed. Resources freed.")


@app.post("/api/chat/stream")
async def chat_stream(request: Request, body: ChatRequest):
    return StreamingResponse(
        event_generator(request, body),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
