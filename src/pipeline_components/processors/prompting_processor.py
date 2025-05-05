import asyncio
from src.pipeline_components.clients.openai_client import call_llm as call_llm_openai
from src.pipeline_components.clients.llamaapi_client import call_llm as call_llm_llama
from src.pipeline_components.clients.gemini_client import call_llm as call_llm_gemini
from src.pipeline_components.clients.grok_client import call_llm as call_llm_grok
from src.pipeline_components.clients.anthropic_client import call_llm as call_llm_anthropic
import traceback
import os
import pandas as pd
from tqdm import tqdm


async def run_prompt(idx: int, row: pd.Series, model: str) -> dict:
    prompt_language1 = row['prompt_language1']
    prompt_language2 = row['prompt_language2']
    prompt_baseline = row['prompt_baseline']
    if model.startswith("gpt"):
        response_language1 = await call_llm_openai(prompt_language1, model=model)
        response_language2 = await call_llm_openai(prompt_language2, model=model)
        response_baseline = await call_llm_openai(prompt_baseline, model=model)
    elif model.startswith("google"):
        response_language1 = await call_llm_gemini(prompt_language1, model=model)
        response_language2 = await call_llm_gemini(prompt_language2, model=model)
        response_baseline = await call_llm_gemini(prompt_baseline, model=model)
    elif model.startswith("grok"):
        response_language1 = await call_llm_grok(prompt_language1, model=model)
        response_language2 = await call_llm_grok(prompt_language2, model=model)
        response_baseline = await call_llm_grok(prompt_baseline, model=model)
    elif model.startswith("claude"):
        response_language1 = await call_llm_anthropic(prompt_language1, model=model)
        response_language2 = await call_llm_anthropic(prompt_language2, model=model)
        response_baseline = await call_llm_anthropic(prompt_baseline, model=model)
    else:
        response_language1 = await call_llm_llama(prompt_language1, model=model)
        response_language2 = await call_llm_llama(prompt_language2, model=model)
        response_baseline = await call_llm_llama(prompt_baseline, model=model)

    return {
        'idx': idx,
        'statement': row['statement'],
        'language1': row['language1'],
        'language2': row['language2'],
        'prompt_language1': prompt_language1,
        'prompt_language2': prompt_language2,
        'prompt_baseline': prompt_baseline,
        'response_language1': response_language1,
        'response_language2': response_language2,
        'response_baseline': response_baseline,
        'statement_embedding': row['statement_embedding'],
    }


async def run_prompts(
        df: pd.DataFrame,
        model: str = "gpt-3.5-turbo",
        max_concurrency: int = 5,
        checkpoint_path: str = 'prompting_checkpoint.pkl',
) -> pd.DataFrame:
    if os.path.exists(checkpoint_path):
        done = pd.read_pickle(checkpoint_path)
        done_idxs = set(done['idx'].tolist())
        to_run = df[~df.index.isin(done_idxs)]
    else:
        done = pd.DataFrame()
        to_run = df

    semaphore = asyncio.Semaphore(max_concurrency)

    async def sem_task(idx, row):
        async with semaphore:
            return await run_prompt(idx, row, model)

    tasks = [
        asyncio.create_task(sem_task(idx, row))
        for idx, row in to_run.iterrows()
    ]

    results = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        try:
            result = await task
            results.append(result)
        except Exception as e:
            traceback.print_exc()
            print(f"Error processing task {task}: {e}")
            break

    new_df = pd.DataFrame(results)
    combined = pd.concat([done, new_df], ignore_index=True)
    combined.to_pickle(checkpoint_path)
    combined = combined.drop(columns=['idx'])
    return combined
