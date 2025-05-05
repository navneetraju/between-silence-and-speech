import os

import pandas as pd
import asyncio
from tqdm import tqdm
from src.pipeline_components.clients.openai_client import get_embedding


def _set_response_embeddings(df: pd.DataFrame, results: list) -> pd.DataFrame:
    if not {'translated_response_language1_embedding',
            'translated_response_language2_embedding', 'response_baseline_embedding'}.issubset(df.columns):
        df['translated_response_language1_embedding'] = None
        df['translated_response_language2_embedding'] = None
        df['response_baseline_embedding'] = None
    for item in results:
        df.at[item['idx'], 'translated_response_language1_embedding'] = item.get(
            'translated_response_language1_embedding')
        df.at[item['idx'], 'translated_response_language2_embedding'] = item.get(
            'translated_response_language2_embedding')
        df.at[item['idx'], 'response_baseline_embedding'] = item.get('response_baseline_embedding')
    return df


async def generate_embeddings(
        df: pd.DataFrame,
        max_concurrency: int = 5,
        checkpoint_path: str = 'embedding_checkpoint.pkl',
) -> pd.DataFrame:
    if os.path.exists(checkpoint_path):
        checkpoint = pd.read_pickle(checkpoint_path)
        done_idxs = set(checkpoint['idx'])
    else:
        checkpoint = pd.DataFrame(columns=['idx'])
        done_idxs = set()

    if 'statement_embedding' not in df.columns:
        df['statement_embedding'] = None

    semaphore = asyncio.Semaphore(max_concurrency)

    async def sem_task(idx, statement):
        async with semaphore:
            embedding = await get_embedding(statement)
            return idx, embedding

    tasks = [
        asyncio.create_task(sem_task(idx, row['statement']))
        for idx, row in df.iterrows() if idx not in done_idxs
    ]

    results = []
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        idx, embedding = await future
        results.append({'idx': idx, 'statement_embedding': embedding})

    for item in results:
        df.at[item['idx'], 'statement_embedding'] = item['statement_embedding']

    if results:
        new_checkpoint = pd.concat([checkpoint, pd.DataFrame(results)], ignore_index=True)
        new_checkpoint.to_pickle(checkpoint_path)

    return df


async def generate_response_embeddings(
        df: pd.DataFrame,
        max_concurrency: int = 5,
        checkpoint_path: str = 'response_embedding_checkpoint.pkl',
) -> pd.DataFrame:
    if os.path.exists(checkpoint_path):
        checkpoint = pd.read_pickle(checkpoint_path)
        done_idxs = set(checkpoint['idx'])
    else:
        checkpoint = pd.DataFrame(columns=['idx'])
        done_idxs = set()

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _get_embedding_set(row):
        language1_embedding = await get_embedding(row['translated_response_language1'])
        language2_embedding = await get_embedding(row['translated_response_language2'])
        baseline_embedding = await get_embedding(row['response_baseline'])
        return language1_embedding, language2_embedding, baseline_embedding

    async def sem_task(idx, row):
        async with semaphore:
            language1_embedding, language2_embedding, baseline_embedding = await _get_embedding_set(row)
            return idx, language1_embedding, language2_embedding, baseline_embedding

    tasks = [
        asyncio.create_task(sem_task(idx, row))
        for idx, row in df.iterrows() if idx not in done_idxs
    ]

    results = []
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        try:
            idx, language1_embedding, language2_embedding, baseline_embedding = await future
            results.append({'idx': idx,
                            'translated_response_language1_embedding': language1_embedding,
                            'translated_response_language2_embedding': language2_embedding,
                            'response_baseline_embedding': baseline_embedding})
        except Exception as e:
            new_df = _set_response_embeddings(df, results)
            if len(checkpoint) > 0:
                new_checkpoint = pd.concat([checkpoint, new_df], ignore_index=True)
            else:
                new_checkpoint = new_df
            new_checkpoint.to_pickle(checkpoint_path)
            raise e

    new_df = _set_response_embeddings(df, results)

    combined = pd.concat([checkpoint, new_df], ignore_index=True)
    combined.to_pickle(checkpoint_path)
    combined = combined.drop(columns=['idx'])
    return combined
