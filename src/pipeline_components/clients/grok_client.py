import asyncio
from joblib import Memory
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI
from src.pipeline_components.config import XAI_API_KEY

memory = Memory(".cache/grok", verbose=0)

_client = OpenAI(
    base_url="https://api.x.ai/v1",
    api_key=XAI_API_KEY,
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
        model: str = "grok-3-beta"
) -> str:
    resp = _client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content


async def call_llm(
        prompt: str,
        model: str = "grok-3-beta"
) -> str:
    return await asyncio.to_thread(_sync_chat_completion, prompt, model)
