import asyncio
from joblib import Memory
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import anthropic
from src.pipeline_components.config import ANTHROPIC_API_KEY

memory = Memory(".cache/anthropic", verbose=0)

_client = anthropic.Anthropic(
    api_key=ANTHROPIC_API_KEY,
)


@memory.cache
@retry(
    reraise=True,
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
)
@memory.cache
@retry(
    reraise=True,
    retry=retry_if_exception_type(Exception),  # retry on any exception
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=1, max=30),
)
def _sync_chat_completion(
        prompt: str,
        model: str = "google/gemini-2.0-flash-001"
) -> str:
    resp = _client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return resp.content[0].text


async def call_llm(
        prompt: str,
        model: str = "google/gemini-2.0-flash-001"
) -> str:
    return await asyncio.to_thread(_sync_chat_completion, prompt, model)
