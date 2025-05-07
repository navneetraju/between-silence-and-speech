from __future__ import annotations

import asyncio

from joblib import Memory
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from google import genai
from google.genai import types
from google.auth import default as google_auth_default
import google.auth.transport.requests

from src.pipeline_components.config import (
    GOOGLE_PROJECT_ID,
    GOOGLE_PROJECT_LOCATION,
    GEMINI_ENDPOINT_ID,
)

memory = Memory(".cache/vertexai", verbose=0)

_credentials, _ = google_auth_default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
_credentials.refresh(google.auth.transport.requests.Request())

_client = genai.Client(
    vertexai=True,
    project=GOOGLE_PROJECT_ID,
    location=GOOGLE_PROJECT_LOCATION,
    credentials=_credentials,
)

_SAFETY_OFF = [
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
]


def _build_config(
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_output_tokens: int = 8192,
) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        response_modalities=["TEXT"],
        safety_settings=_SAFETY_OFF,
    )


@memory.cache
@retry(
    reraise=True,
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
)
def _sync_chat_completion(
        prompt: str,
        *,
        model: str = GEMINI_ENDPOINT_ID,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_output_tokens: int = 8192,
) -> str:
    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
    config = _build_config(temperature, top_p, max_output_tokens)

    response = _client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    return response.text  # -> str


async def call_llm(
        prompt: str,
        *,
        model: str = GEMINI_ENDPOINT_ID,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_output_tokens: int = 8192,
) -> str:
    return await asyncio.to_thread(
        _sync_chat_completion,
        prompt,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
    )
