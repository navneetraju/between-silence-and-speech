import asyncio
import os
import pandas as pd
from googletrans import Translator
from joblib import Memory
import traceback
from tqdm import tqdm

lang_map = {
    'Albanian': 'sq', 'Amharic': 'am', 'Arabic': 'ar', 'Armenian': 'hy',
    'Azerbaijani': 'az', 'Bengali': 'bn', 'Bosnian': 'bs', 'Burmese': 'my',
    'Catalan': 'ca', 'Croatian': 'hr', 'Czech': 'cs', 'Danish': 'da',
    'Dutch': 'nl', 'English': 'en', 'Filipino': 'tl', 'Finnish': 'fi',
    'French': 'fr', 'German': 'de', 'Greek': 'el', 'Haitian Creole': 'ht',
    'Hebrew': 'he', 'Hindi': 'hi', 'Igbo': 'ig', 'Indonesian': 'id',
    'Irish': 'ga', 'Italian': 'it', 'Japanese': 'ja', 'Kazakh': 'kk',
    'Khmer': 'km', 'Kinyarwanda': 'rw', 'Latvian': 'lv', 'Macedonian': 'mk',
    'Mandarin': 'zh-cn', 'Mandarin Chinese': 'zh-cn', 'Nepali': 'ne',
    'Pashto': 'ps', 'Persian': 'fa', 'Polish': 'pl', 'Portuguese': 'pt',
    'Romanian': 'ro', 'Russian': 'ru', 'Serbian': 'sr', 'Shona': 'sn',
    'Sinhala': 'si', 'Somali': 'so', 'Spanish': 'es', 'Swahili': 'sw',
    'Swedish': 'sv', 'Thai': 'th', 'Tigrinya': 'ti', 'Turkish': 'tr',
    'Ukrainian': 'uk', 'Urdu': 'ur', 'Vietnamese': 'vi', 'Zulu': 'zu'
}

memory = Memory(location=".cache/statement_translation", verbose=0)


@memory.cache
def _cached_translate(text: str, src: str, dest: str) -> str:
    async def _do_translate():
        translator = Translator()
        tr = await translator.translate(text, src=src, dest=dest)
        return tr.text

    return asyncio.run(_do_translate())


async def translate_text(text: str, src_full: str, dest_full: str) -> str:
    if not isinstance(src_full, str) or not isinstance(dest_full, str):
        return text
    src = lang_map.get(src_full, src_full)
    dest = lang_map.get(dest_full, dest_full)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _cached_translate, text, src, dest)


async def process_row(idx, row):
    prompt_language1, prompt_language2 = row['prompt_language1'], row['prompt_language2']
    l1, l2 = row['language1'], row['language2']
    t1 = asyncio.create_task(translate_text(prompt_language1, "English", l1))
    t2 = asyncio.create_task(translate_text(prompt_language2, "English", l2))
    out1, out2 = await asyncio.gather(t1, t2)
    return {
        'idx': idx,
        'statement': row['statement'],
        'language1': l1,
        'language2': l2,
        'prompt_language1': out1,
        'prompt_language2': out2,
        'prompt_baseline': row['prompt_baseline'],
        'statement_embedding': row['statement_embedding'],
    }


async def process_response_rows(idx, row):
    language1, language2 = row['language1'], row['language2']
    if language1 != 'English':
        out1 = asyncio.run(translate_text(row['response_language1'], language1, "English"))
    else:
        out1 = row['response_language1']
    if language2 != 'English':
        out2 = asyncio.run(translate_text(row['response_language2'], language2, "English"))
    else:
        out2 = row['response_language2']
    return {
        'idx': idx,
        'model_name': row['model_name'],
        'statement': row['statement'],
        'language1': language1,
        'language2': language2,
        'prompt_language1': row['prompt_language1'],
        'prompt_language2': row['prompt_language2'],
        'prompt_baseline': row['prompt_baseline'],
        'response_language1': row['response_language1'],
        'response_language2': row['response_language2'],
        'response_baseline': row['response_baseline'],
        'translated_response_language1': out1,
        'translated_response_language2': out2,
        'statement_embedding': row['statement_embedding'],
    }


async def translate_dataframe(
        df: pd.DataFrame,
        max_concurrency: int = 5,
        checkpoint_path: str = 'translation_checkpoint.pkl',
        process_responses: bool = False,
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
            if process_responses:
                return await process_response_rows(idx, row)
            return await process_row(idx, row)

    tasks = [
        asyncio.create_task(sem_task(idx, row))
        for idx, row in to_run.iterrows()
    ]

    results = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        try:
            res = await task
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Error processing a row: {e}")
        results.append(res)

    new_df = pd.DataFrame(results)
    combined = pd.concat([done, new_df], ignore_index=True)
    combined.to_pickle(checkpoint_path)
    # Remove the idx column
    combined = combined.drop(columns=['idx'])
    return combined
