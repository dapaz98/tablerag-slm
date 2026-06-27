from pathlib import Path

import time
import pandas as pd

from src.text_to_sql.load_wtq_questions import WTQQuestionLoader
from src.text_to_sql.schema_extractor import SchemaExtractor
from src.text_to_sql.prompt_builder import PromptBuilder
from src.text_to_sql.sql_generator import SQLGenerator
from src.text_to_sql.execute_sql import SQLExecutor

from src.retrieval.embedding_retriever import EmbeddingRetriever
from src.text_to_sql.WTQTableLoader import TableContentLoader

import os

from dotenv import load_dotenv

from src.text_to_sql.together_sql_generator import TogetherSQLGenerator

from src.text_to_sql.sql_cleaner import clean_sql

load_dotenv()

# =====================================
# CONFIG
# =====================================

# modelos que rodam via Together AI
TOGETHER_MODELS = [
    # "openai/gpt-oss-120b",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo"
]

# todos os modelos — locais + API
MODELS = [
    # "llama3.2:3b",
    # "deepseek-coder:6.7b",
    # "qwen2.5-coder:7b",
    "gemma4:e4b",
    # "openai/gpt-oss-120b",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo"
]

# nome curto para o CSV
MODEL_DISPLAY_NAMES = {
    "openai/gpt-oss-120b": "gpt-oss-120b",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "llama-3.3-70b",
}

PROMPT_METHODS = [
    "zero-shot",
    "content-aware",
    "embedding-retrieval",
    "embedding-content-aware",
    "schema-linking"
]

MAX_SAMPLES = 1000

DATASET_PATH = (
    "data/raw/WikiTableQuestions/"
    "data/random-split-1-dev.tsv"
)

DATABASE_PATH = (
    "data/databases/wikitablequestions.db"
)

RESULTS_PATH = (
    "results/final_benchmark.csv"
)


# =====================================
# LOADERS
# =====================================

loader = WTQQuestionLoader(DATASET_PATH)
loader.load_dataset()
questions = loader.get_all_questions()

schema_extractor = SchemaExtractor(DATABASE_PATH)
retriever = EmbeddingRetriever()
table_loader = TableContentLoader(DATABASE_PATH)
prompt_builder = PromptBuilder()
sql_executor = SQLExecutor(DATABASE_PATH)


# =====================================
# CARREGA RESULTADOS ANTERIORES
# =====================================

results_path = Path(RESULTS_PATH)

GROQ_ERROR = "Groq esgotou 5 tentativas sem resposta válida."

if results_path.exists():

    existing_df = pd.read_csv(RESULTS_PATH)

    failed_df = existing_df[
        existing_df["execution_error"] == GROQ_ERROR
    ]

    successful_df = existing_df[
        existing_df["execution_error"] != GROQ_ERROR
    ]

    results = successful_df.to_dict("records")

    processed = set(
        zip(
            successful_df["model_name"],
            successful_df["prompt_method"],
            successful_df["question_id"]
        )
    )

    print(
        f"[RESUMINDO] "
        f"{len(processed)} já processadas, "
        f"{len(failed_df)} para reprocessar."
    )

else:

    results = []
    processed = set()


# =====================================
# HELPER: SALVA INCREMENTALMENTE
# =====================================

def save_results(results, path):
    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(path, index=False)


# =====================================
# MODEL LOOP
# =====================================

for model_name in MODELS:

    # nome curto para o CSV
    display_name = MODEL_DISPLAY_NAMES.get(
        model_name, model_name
    )

    print("\n======================")
    print(f"MODEL: {display_name}")
    print("======================")

    if model_name in TOGETHER_MODELS:
        sql_generator = TogetherSQLGenerator(
            api_key=os.getenv("TOGETHER_API_KEY"),
            model_name=model_name
        )
    else:
        sql_generator = SQLGenerator(model_name)


    # =================================
    # PROMPT LOOP
    # =================================

    for prompt_method in PROMPT_METHODS:

        print("\n----------------------")
        print(f"PROMPT: {prompt_method}")
        print("----------------------")


        # =============================
        # QUESTION LOOP
        # =============================

        for sample in questions[:MAX_SAMPLES]:

            generated_sql = None

            # ==========================
            # SKIP SE JÁ PROCESSADO
            # usa display_name para consistência
            # ==========================

            key = (
                display_name,
                prompt_method,
                sample["question_id"]
            )

            if key in processed:
                print(f"[SKIP] {sample['question_id']}")
                continue

            try:

                question_id = sample["question_id"]
                question    = sample["question"]
                table_name  = sample["table_name"]
                gold_answer = str(sample["gold_answer"]).strip()

                columns = (
                    schema_extractor
                    .get_table_schema(table_name)
                )

                retrieved_rows = None

                # ======================
                # BUILD PROMPT
                # ======================

                if prompt_method == "zero-shot":

                    prompt = prompt_builder.build_zero_shot_prompt(
                        question, table_name, columns
                    )

                elif prompt_method == "few-shot":

                    prompt = prompt_builder.build_few_shot_prompt(
                        question, table_name, columns
                    )

                elif prompt_method == "cot":

                    prompt = prompt_builder.build_cot_prompt(
                        question, table_name, columns
                    )

                elif prompt_method == "content-aware":

                    sample_rows = schema_extractor.get_sample_rows(
                        table_name, limit=3
                    )

                    prompt = prompt_builder.build_content_aware_prompt(
                        question, table_name, columns, sample_rows
                    )

                elif prompt_method == "embedding-retrieval":

                    rows_df = table_loader.get_table_rows(table_name)

                    rows_text = [
                        " | ".join(row.astype(str))
                        for _, row in rows_df.iterrows()
                    ]

                    top_rows = retriever.retrieve_top_k_rows(
                        question, rows_text, k=5
                    )

                    retrieved_rows = top_rows

                    prompt = prompt_builder.build_embedding_prompt(
                        question, table_name, columns, top_rows
                    )

                elif prompt_method == "embedding-content-aware":

                    rows_df = table_loader.get_table_rows(table_name)

                    rows_text = [
                        " | ".join(row.astype(str))
                        for _, row in rows_df.iterrows()
                    ]

                    top_rows = retriever.retrieve_top_k_rows(
                        question, rows_text, k=5
                    )

                    retrieved_rows = top_rows

                    prompt = prompt_builder.build_embedding_content_aware_prompt(
                        question, table_name, columns, top_rows
                    )

                elif prompt_method == "schema-linking":

                    rows_df = table_loader.get_table_rows(table_name)

                    rows_text = [
                        " | ".join(row.astype(str))
                        for _, row in rows_df.iterrows()
                    ]

                    top_rows = retriever.retrieve_top_k_rows(
                        question, rows_text, k=5
                    )

                    retrieved_rows = top_rows

                    prompt = prompt_builder.build_schema_linking_prompt(
                        question, table_name, columns, top_rows,
                        top_k_columns=5
                    )

                else:

                    raise ValueError("Invalid PROMPT_METHOD.")

                # ======================
                # PROMPT METRICS
                # ======================

                prompt_tokens = len(prompt.split())

                # ======================
                # GENERATE SQL
                # ======================

                generation_start = time.time()

                generated_sql = sql_generator.generate_sql(prompt)

                generation_end = time.time()

                latency_seconds = generation_end - generation_start

                # ======================
                # CLEAN SQL
                # ======================

                generated_sql = clean_sql(generated_sql)

                sql_generated = (
                    generated_sql is not None
                    and generated_sql != ""
                )

                # ======================
                # EXECUTE SQL
                # ======================

                execution_success = True
                has_result        = False
                execution_error   = None
                result            = None

                try:

                    result = sql_executor.execute_query(generated_sql)

                    if result is not None and len(result) > 0:
                        has_result = True

                except Exception as e:

                    execution_success = False
                    execution_error   = str(e)

                    print(f"[EXECUTION ERROR] {e}")

                # ======================
                # PREDICTED ANSWER
                # ======================

                predicted_answer = None

                if has_result:
                    try:
                        predicted_answer = str(
                            result.iloc[0, 0]
                        ).strip()
                    except Exception:
                        predicted_answer = None

                # ======================
                # FINAL EVALUATION
                # ======================

                is_correct = (
                    str(predicted_answer).strip().lower()
                    ==
                    str(gold_answer).strip().lower()
                )

                # ======================
                # SAVE RESULT
                # ======================

                experiment = {
                    "question_id"      : question_id,
                    "model_name"       : display_name,  # nome curto
                    "prompt_method"    : prompt_method,
                    "question"         : question,
                    "table_name"       : table_name,
                    "columns"          : str(columns),
                    "prompt"           : prompt,
                    "generated_sql"    : generated_sql,
                    "predicted_answer" : predicted_answer,
                    "gold_answer"      : gold_answer,
                    "execution_success": execution_success,
                    "has_result"       : has_result,
                    "correct"          : is_correct,
                    "latency_seconds"  : latency_seconds,
                    "execution_error"  : execution_error,
                    "prompt_tokens"    : prompt_tokens,
                    "retrieved_rows"   : str(retrieved_rows),
                    "sql_generated"    : sql_generated,
                }

                results.append(experiment)
                save_results(results, RESULTS_PATH)

                print(
                    f"[OK] {question_id} "
                    f"| Model: {display_name} "
                    f"| Prompt: {prompt_method} "
                    f"| Exec: {execution_success} "
                    f"| Result: {has_result} "
                    f"| Correct: {is_correct}"
                )

            except Exception as e:

                print(f"[PIPELINE ERROR] {e}")

                results.append({
                    "question_id"      : sample.get("question_id"),
                    "model_name"       : display_name,  # nome curto
                    "prompt_method"    : prompt_method,
                    "question"         : sample.get("question"),
                    "table_name"       : sample.get("table_name"),
                    "columns"          : None,
                    "prompt"           : None,
                    "generated_sql"    : generated_sql,
                    "predicted_answer" : None,
                    "gold_answer"      : str(sample.get("gold_answer", "")).strip(),
                    "execution_success": False,
                    "has_result"       : False,
                    "correct"          : False,
                    "latency_seconds"  : None,
                    "execution_error"  : str(e),
                    "prompt_tokens"    : None,
                    "retrieved_rows"   : None,
                    "sql_generated"    : False,
                })

                save_results(results, RESULTS_PATH)


# =====================================
# FINAL SUMMARY
# =====================================

results_dataframe = pd.read_csv(RESULTS_PATH)

if results_dataframe.empty:

    print("\n======================")
    print("NO RESULTS GENERATED")
    print("======================")

else:

    print("\n======================")
    print("FINAL RESULTS")
    print("======================")

    summary = (
        results_dataframe
        .groupby(["model_name", "prompt_method"])
        .agg({
            "correct"          : "mean",
            "sql_generated"    : "mean",
            "execution_success": "mean",
            "has_result"       : "mean",
            "prompt_tokens"    : "mean",
            "latency_seconds"  : "mean"
        })
        .reset_index()
    )

    print(summary)

    print("\n======================")
    print("TOP EXECUTION ERRORS")
    print("======================")

    print(
        results_dataframe["execution_error"]
        .dropna()
        .value_counts()
        .head(10)
    )

    print("\n======================")
    print("CSV SAVED")
    print("======================")

    print(RESULTS_PATH)