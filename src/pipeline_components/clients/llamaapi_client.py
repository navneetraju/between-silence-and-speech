import asyncio
import json
from joblib import Memory
from llamaapi import LlamaAPI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.pipeline_components.config import LLAMA_API_KEY

memory = Memory(".cache/llamaapi", verbose=0)
_client = LlamaAPI(LLAMA_API_KEY)


@memory.cache
@retry(
    reraise=True,
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=1, max=30),
)
def _sync_chat_completion(
        prompt: str,
        model: str = "llama-3",
) -> str:
    api_request_json = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "temperature": 0.7,
    }

    response = _client.run(api_request_json)
    try:
        response_json = response.json()
        if "generated_text" in response_json:
            return response_json["generated_text"]
        elif "choices" in response_json:
            return response_json["choices"][0]["message"]["content"]
        else:
            return json.dumps(response_json)
    except Exception as e:
        print(f"Error parsing response for prompt: {prompt[:30]}... : {e}")
        raise RuntimeError(
            f"Error parsing response for prompt: {prompt[:30]}... : {e}"
        ) from e


async def call_llm(
        prompt: str,
        model: str = "llama-3",
) -> str:
    return await asyncio.to_thread(_sync_chat_completion, prompt, model)
