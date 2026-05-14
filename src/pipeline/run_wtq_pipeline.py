from pathlib import Path

import time
import pandas as pd

from src.text_to_sql.load_wtq_questions import WTQQuestionLoader
from src.text_to_sql.schema_extractor import SchemaExtractor
from src.text_to_sql.prompt_builder import PromptBuilder
from src.text_to_sql.sql_generator import SQLGenerator
from src.text_to_sql.execute_sql import SQLExecutor


# =====================================
# CONFIG
# =====================================

MODEL_NAME = "gemma4:e4b"

PROMPT_METHOD = "zero-shot"
# "few-shot"
# "cot"

MAX_SAMPLES = 10


DATASET_PATH = (
    "data/raw/WikiTableQuestions/"
    "data/random-split-1-dev.tsv"
)

DATABASE_PATH = (
    "data/databases/wikitablequestions.db"
)

RESULTS_PATH = (
    f"results/"
    f"{MODEL_NAME.replace(':', '_')}_"
    f"{PROMPT_METHOD}.csv"
)


# =====================================
# LOADERS
# =====================================

loader = WTQQuestionLoader(
    DATASET_PATH
)

loader.load_dataset()

schema_extractor = SchemaExtractor(
    DATABASE_PATH
)

prompt_builder = PromptBuilder()

sql_generator = SQLGenerator(
    MODEL_NAME
)

sql_executor = SQLExecutor(
    DATABASE_PATH
)


# =====================================
# RUN PIPELINE
# =====================================

results = []

questions = loader.get_all_questions()

for sample in questions[:MAX_SAMPLES]:

    try:

        # ==============================
        # LOAD SAMPLE
        # ==============================

        question_id = sample["question_id"]

        question = sample["question"]

        table_name = sample["table_name"]

        gold_answer = str(
            sample["gold_answer"]
        ).strip()


        # ==============================
        # EXTRACT SCHEMA
        # ==============================

        columns = schema_extractor.get_table_schema(
            table_name
        )


        # ==============================
        # BUILD PROMPT
        # ==============================

        if PROMPT_METHOD == "zero-shot":

            prompt = (
                prompt_builder
                .build_zero_shot_prompt(
                    question,
                    table_name,
                    columns
                )
            )

        elif PROMPT_METHOD == "few-shot":

            prompt = (
                prompt_builder
                .build_few_shot_prompt(
                    question,
                    table_name,
                    columns
                )
            )

        elif PROMPT_METHOD == "cot":

            prompt = (
                prompt_builder
                .build_cot_prompt(
                    question,
                    table_name,
                    columns
                )
            )

        else:

            raise ValueError(
                "Invalid PROMPT_METHOD."
            )


        # ==============================
        # GENERATE SQL
        # ==============================

        generation_start = time.time()

        generated_sql = (
            sql_generator.generate_sql(
                prompt
            )
        )

        generation_end = time.time()

        latency_seconds = (
            generation_end - generation_start
        )


        # ==============================
        # EXECUTE SQL
        # ==============================

        execution_success = True

        try:

            result = sql_executor.execute_query(
                generated_sql
            )

        except Exception as execution_error:

            execution_success = False

            result = None

            print(
                f"[EXECUTION ERROR] "
                f"{execution_error}"
            )


        # ==============================
        # PREDICTED ANSWER
        # ==============================

        try:

            predicted_answer = str(
                result.iloc[0, 0]
            ).strip()

        except Exception:

            predicted_answer = None


        # ==============================
        # FINAL EVALUATION
        # ==============================

        is_correct = (
            predicted_answer == gold_answer
        )


        # ==============================
        # SAVE RESULT
        # ==============================

        experiment = {

            # --------------------------
            # METADATA
            # --------------------------

            "question_id": question_id,

            "model_name": MODEL_NAME,

            "prompt_method": PROMPT_METHOD,


            # --------------------------
            # INPUT
            # --------------------------

            "question": question,

            "table_name": table_name,

            "columns": str(columns),

            "prompt": prompt,


            # --------------------------
            # GENERATED OUTPUT
            # --------------------------

            "generated_sql": generated_sql,

            "predicted_answer": predicted_answer,


            # --------------------------
            # GOLD
            # --------------------------

            "gold_answer": gold_answer,


            # --------------------------
            # METRICS
            # --------------------------

            "execution_success": execution_success,

            "correct": is_correct,

            "latency_seconds": latency_seconds
        }

        results.append(experiment)


        # ==============================
        # LOG
        # ==============================

        print(
            f"[OK] "
            f"{question_id} "
            f"| Correct: {is_correct} "
            f"| Execution: {execution_success}"
        )

    except Exception as e:

        print(
            f"[PIPELINE ERROR] "
            f"{e}"
        )


# =====================================
# SAVE RESULTS
# =====================================

results_dataframe = pd.DataFrame(
    results
)

Path("results").mkdir(
    exist_ok=True
)

results_dataframe.to_csv(
    RESULTS_PATH,
    index=False
)


# =====================================
# FINAL SUMMARY
# =====================================

accuracy = (
    results_dataframe["correct"]
    .mean()
)

execution_accuracy = (
    results_dataframe["execution_success"]
    .mean()
)

average_latency = (
    results_dataframe["latency_seconds"]
    .mean()
)

print("\n======================")
print("FINAL RESULTS")
print("======================")

print(f"Model: {MODEL_NAME}")

print(f"Prompt Method: {PROMPT_METHOD}")

print(f"Samples: {MAX_SAMPLES}")

print(f"Denotation Accuracy: {accuracy:.4f}")

print(
    f"Execution Accuracy: "
    f"{execution_accuracy:.4f}"
)

print(
    f"Average Latency: "
    f"{average_latency:.2f}s"
)

print("\n======================")
print("CSV SAVED")
print("======================")

print(RESULTS_PATH)