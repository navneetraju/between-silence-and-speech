import asyncio
from joblib import Memory
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI, BadRequestError
from src.pipeline_components.config import OPENAI_API_KEY

memory = Memory(".cache/openai", verbose=0)

_client = OpenAI(api_key=OPENAI_API_KEY)


@memory.cache
@retry(
    reraise=True,
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
)
def _sync_get_embedding(text: str, model: str = "text-embedding-3-large") -> list[float]:
    if len(text) == 0:
        print(f"Warning: Empty text provided. Returning empty embedding.")
        return []
    if len(text.split(" ")) > 8192:
        print(f"Warning: Text length exceeds 8192 tokens. Truncating to first 8192 tokens.")
    max_token_text = text[:8192]
    try:
        resp = _client.embeddings.create(input=[max_token_text], model=model)
    except BadRequestError as e:
        print(f"BadRequestError: The text is too long or the model is invalid.")
        print(f"Error details: {e}")
        print(f"Text: {text}")
        print(f"Trunc Text: {max_token_text}")
        raise e
    return resp.data[0].embedding


async def get_embedding(
        text: str,
        model: str = "text-embedding-3-large"
) -> list[float]:
    return await asyncio.to_thread(_sync_get_embedding, text, model)


@memory.cache
@retry(
    reraise=True,
    retry=retry_if_exception_type(Exception),  # retry on any exception
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=1, max=30),
)
def _sync_chat_completion(
        prompt: str,
        model: str = "gpt-3.5-turbo"
) -> str:
    resp = _client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content


async def call_llm(
        prompt: str,
        model: str = "gpt-3.5-turbo"
) -> str:
    return await asyncio.to_thread(_sync_chat_completion, prompt, model)
