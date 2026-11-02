import os

import httpx


OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:4b"
)


async def generate_text(prompt: str) -> str:
    """
    Send a prompt to the local Ollama model and return its response.
    """

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2
        }
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload
        )

        response.raise_for_status()

        result = response.json()

    generated_text = result.get("response", "").strip()

    if not generated_text:
        raise RuntimeError(
            "Ollama returned an empty response."
        )

    return generated_text