import asyncio
from joblib import Memory
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI
from google.auth import default
import google.auth.transport.requests
from src.pipeline_components.config import GOOGLE_PROJECT_LOCATION, GOOGLE_PROJECT_ID

memory = Memory(".cache/gemini", verbose=0)

credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
credentials.refresh(google.auth.transport.requests.Request())

_client = OpenAI(
    base_url=f"https://{GOOGLE_PROJECT_LOCATION}-aiplatform.googleapis.com/v1/projects/{GOOGLE_PROJECT_ID}/locations/{GOOGLE_PROJECT_LOCATION}/endpoints/openapi",
    api_key=credentials.token,
)


@memory.cache
@retry(
    reraise=True,
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
)
def _sync_get_embedding(text: str, model: str = "text-embedding-3-large") -> list[float]:
    resp = _client.embeddings.create(input=[text], model=model)
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
        model: str = "google/gemini-2.0-flash-001"
) -> str:
    resp = _client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content


async def call_llm(
        prompt: str,
        model: str = "google/gemini-2.0-flash-001"
) -> str:
    return await asyncio.to_thread(_sync_chat_completion, prompt, model)
