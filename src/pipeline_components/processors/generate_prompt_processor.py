import asyncio

import jinja2
from importlib import resources
import os
from tqdm import tqdm

import pandas as pd


def load_prompt_template() -> jinja2.Template:
    text = resources.read_text(
        'src.pipeline_components.resources',
        f'llm_few_shot_prompt.j2'
    )
    return jinja2.Template(text)


async def generate_prompt(
        df: pd.DataFrame,
        max_concurrency: int = 5,
        checkpoint_path: str = 'prompt_templating_checkpoint.pkl',
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
            template = load_prompt_template()
            prompt_language1 = template.render(
                statement=row['statement'],
                language=row['language1'],
            )
            prompt_language2 = template.render(
                statement=row['statement'],
                language=row['language2'],
            )
            prompt_baseline = template.render(
                statement=row['statement'],
                language='English',
            )
            return {
                'idx': idx,
                'statement': row['statement'],
                'language1': row['language1'],
                'language2': row['language2'],
                'prompt_language1': prompt_language1,
                'prompt_language2': prompt_language2,
                'prompt_baseline': prompt_baseline
            }

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
    # Remove the idx column
    combined = combined.drop(columns=['idx'])
    return combined
