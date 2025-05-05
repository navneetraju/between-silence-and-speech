import asyncio
import os

import pandas as pd
from scipy.spatial.distance import cosine
from tqdm import tqdm


def process_row(idx, row):
    if row['translated_response_language1_embedding'] is None or len(
            row['translated_response_language1_embedding']) == 0:
        print(f"Row {idx} has empty response_language1_embedding")
        row['translated_response_language1_embedding'] = [0] * 3072
    if row['translated_response_language2_embedding'] is None or len(
            row['translated_response_language2_embedding']) == 0:
        print(f"Row {idx} has empty response_language2_embedding")
        row['translated_response_language2_embedding'] = [0] * 3072
    if row['response_baseline_embedding'] is None or len(row['response_baseline_embedding']) == 0:
        print(f"Row {idx} has empty response_baseline_embedding")
        row['response_baseline_embedding'] = [0] * 3072
    dist_prompt_lang1 = cosine(
        row['translated_response_language1_embedding'],
        row['statement_embedding']
    )
    dist_prompt_lang2 = cosine(
        row['translated_response_language2_embedding'],
        row['statement_embedding']
    )
    dist_baseline = cosine(
        row['response_baseline_embedding'],
        row['statement_embedding']
    )
    return {
        'idx': idx,
        **row.to_dict(),
        'translated_response_lang1_distance': dist_prompt_lang1,
        'translated_response_lang2_distance': dist_prompt_lang2,
        'response_baseline_distance': dist_baseline,
    }


async def generate_distance(
        df: pd.DataFrame,
        max_concurrency: int = 5,
        checkpoint_path: str = 'distance_checkpoint.pkl',
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
            return await asyncio.to_thread(process_row, idx, row)

    tasks = [
        asyncio.create_task(sem_task(idx, row))
        for idx, row in to_run.iterrows()
    ]

    results = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await task
        results.append(result)

    new_df = pd.DataFrame(results)
    combined = pd.concat([done, new_df], ignore_index=True)
    combined.to_pickle(checkpoint_path)
    combined = combined.drop(columns=['idx'])
    return combined
