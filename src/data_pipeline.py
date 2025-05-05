import asyncio
from src.pipeline_components.config import LLM_API_MODEL_LIST, LLM_PROMPTING_CONCURRENCY

import typer
import pandas as pd
from src.pipeline_components.processors.generate_prompt_processor import generate_prompt
from src.pipeline_components.processors.translation_processor import translate_dataframe
from src.pipeline_components.processors.embedding_processor import generate_embeddings, generate_response_embeddings
from src.pipeline_components.processors.prompting_processor import run_prompts
from src.pipeline_components.processors.spacy_processor import generate_metrics
from src.pipeline_components.processors.cos_distance_processor import generate_distance

app = typer.Typer()


@app.command()
def main(
        data_file_path: str = typer.Option(
            "./data/sample_data.csv", help="Path to the csv data file."
        ),
        output_file_path: str = typer.Option(
            "./data/sample_output.parquet", help="Path to the output parquet file."
        )
):
    print(f"[INFO] Reading data from {data_file_path}...")
    df = pd.read_csv(data_file_path)

    # Step 1: Setup the prompts
    print("\n[Step 1] Setting up few-shot prompts...")
    df = asyncio.run(
        generate_prompt(
            df,
            max_concurrency=5,
            checkpoint_path='.cache/gen_prompt_checkpoint.pkl'
        )
    )

    # Step 2: Generate embeddings for the statements
    print("\n[Step 2] Generating embeddings for original statements...")
    df = asyncio.run(
        generate_embeddings(
            df,
            max_concurrency=5,
            checkpoint_path='.cache/embedding_checkpoint.pkl'
        )
    )

    # Step 3: Translate the prompts
    print("\n[Step 3] Translating prompts...")
    df = asyncio.run(
        translate_dataframe(
            df,
            max_concurrency=5,
            checkpoint_path='.cache/translation_checkpoint.pkl'
        )
    )

    # Step 4: Prompt LLMs starting with LLAMA API based ones
    print(f"\n[Step 4] Prompting {len(LLM_API_MODEL_LIST)} LLMs...")
    dfs = []
    for model in LLM_API_MODEL_LIST:
        print(f"\n[INFO] Prompting {model}...")
        checkpoint_path = f'.cache/prompting_checkpoint_{model.replace("/", "_")}.pkl'
        df_model = asyncio.run(
            run_prompts(
                df,
                model=model,
                max_concurrency=LLM_PROMPTING_CONCURRENCY.get(model, 5),
                checkpoint_path=checkpoint_path
            )
        )
        dfs.append((model, df_model))

    # create a new column called model_name and then merge the dfs
    merged_df = pd.DataFrame()
    for model, df_model in dfs:
        df_model['model_name'] = model
        merged_df = pd.concat([merged_df, df_model], ignore_index=True)

    # Step 5: Translate the responses back to English
    print("\n[Step 5] Translating responses back to English...")
    merged_df = asyncio.run(
        translate_dataframe(
            merged_df,
            max_concurrency=100,
            checkpoint_path='.cache/translation_checkpoint_responses.pkl',
            process_responses=True
        )
    )

    # Step 6: Generate embeddings for the responses
    print("\n[Step 6] Generating embeddings for responses...")
    merged_df = asyncio.run(
        generate_response_embeddings(
            merged_df,
            max_concurrency=5,
            checkpoint_path='.cache/response_embedding_checkpoint.pkl'
        )
    )

    # Step 7: Generate metrics (polarity, subjectivity, refusal)
    print("\n[Step 7] Generating metrics...")
    merged_df = asyncio.run(
        generate_metrics(
            merged_df,
            max_concurrency=10,
            checkpoint_path='.cache/metrics_checkpoint.pkl'
        )
    )

    # Step 8: Generate cosine distance metrics
    print("\n[Step 8] Generating cosine distance metrics...")
    merged_df = asyncio.run(
        generate_distance(
            merged_df,
            max_concurrency=10,
            checkpoint_path='.cache/distance_checkpoint.pkl'
        )
    )

    # Save the final dataframe to a CSV file
    merged_df.to_parquet(output_file_path, index=False)


if __name__ == '__main__':
    app()
