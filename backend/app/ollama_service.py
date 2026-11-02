import os
import json
from typing import Any

import httpx


OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434"
).rstrip("/")

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:4b"
)

OLLAMA_TIMEOUT = float(
    os.getenv("OLLAMA_TIMEOUT", "180")
)


async def generate_text(prompt: str) -> str:
    """
    Generate a text response using the local Ollama model.

    Kept compatible with the existing /api/llm-test endpoint.
    """

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 400
        }
    }

    async with httpx.AsyncClient(
        timeout=OLLAMA_TIMEOUT
    ) as client:
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


async def generate_json(
    prompt: str,
    schema: dict[str, Any]
) -> dict[str, Any]:
    """
    Generate structured JSON using Ollama's chat API.

    The JSON schema constrains the response format.
    Thinking is disabled where supported to reduce
    unnecessary reasoning output and inference time.
    """

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a contract analysis assistant. "
                    "Analyze only the supplied contract text. "
                    "Do not invent clauses or legal facts. "
                    "Return concise, valid JSON matching "
                    "the supplied schema."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "format": schema,
        "think": False,
        "options": {
            "temperature": 0,
            "num_predict": 300
        }
    }

    async with httpx.AsyncClient(
        timeout=OLLAMA_TIMEOUT
    ) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload
        )

        response.raise_for_status()
        result = response.json()

    message = result.get("message", {})
    content = message.get("content", "").strip()

    if not content:
        raise RuntimeError(
            "Ollama returned an empty structured response."
        )

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Ollama returned invalid JSON."
        ) from exc

    if not isinstance(parsed, dict):
        raise RuntimeError(
            "Ollama's structured response must be a JSON object."
        )

    return parsed