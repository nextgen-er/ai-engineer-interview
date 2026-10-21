import asyncio
import httpx
from mock_data import DEFAULT_TEST_PAYLOAD


async def test_disconnect():
    url = "http://localhost:8000/api/chat/stream"
    payload = {
        **DEFAULT_TEST_PAYLOAD,
        "max_tokens": 100,
        "delay_per_token": 0.1,
    }

    print("Connecting to stream...")
    async with httpx.AsyncClient(timeout=10.0) as client:
        async with client.stream("POST", url, json=payload) as response:
            count = 0
            async for chunk in response.aiter_lines():
                if chunk.strip():
                    print(f"Received chunk {count}: {chunk}")
                    count += 1
                if count >= 5:
                    print("\n---> Deliberately breaking the connection now! <---")
                    break

    print(
        "Client connection closed. Check your server logs to ensure the generator aborted."
    )


if __name__ == "__main__":
    asyncio.run(test_disconnect())
