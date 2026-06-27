import re
import inflect
import pandas as pd

from pathlib import Path

from src.config.settings import ProcessConfig
from src.preprocessing.database import SQLiteManager


# =========================
# CONFIG
# =========================

config = ProcessConfig()

CSV_ROOT = Path(config.csv_wikitq)

DATABASE_PATH = config.database_path

p = inflect.engine()

NORMALIZATION_STATS = {

    "total_columns": 0,

    "renamed_columns": 0,

    "numeric_columns": 0,

    "unnamed_columns": 0,

    "duplicate_columns": 0
}

# =========================
# COLUMN NORMALIZATION
# =========================
def normalize_column_name(column):

    original = str(column)

    column = original.strip()

    NORMALIZATION_STATS["total_columns"] += 1

    # Unnamed columns
    if column.lower().startswith("unnamed"):

        NORMALIZATION_STATS["unnamed_columns"] += 1

        NORMALIZATION_STATS["renamed_columns"] += 1

        return "row_id"

    replacements = {

        "%": " percent ",
        "$": " dollar ",
        "&": " and ",
        "+": "_plus_",
        "/": "_",
        "-": "_",
        "#": "",
        "(": "_",
        ")": "_",
        ".": "_",
        ",": "_",
        ":": "_",
        ";": "_"
    }

    for old, new in replacements.items():

        column = column.replace(
            old,
            new
        )

    column = re.sub(
        r"[^\w]",
        "_",
        column
    )

    column = re.sub(
        r"_+",
        "_",
        column
    )

    column = column.strip("_")

    if re.match(r"^\d", column):

        NORMALIZATION_STATS["numeric_columns"] += 1

        match = re.match(
            r"^(\d+)(.*)",
            column
        )

        number = match.group(1)

        rest = match.group(2)

        number_text = (

            p.number_to_words(number)

            .replace("-", "_")

            .replace(" ", "_")
        )

        column = (
            number_text
            + rest
        )

    column = column.strip("_")

    column = column.lower()

    # se virou vazio após a limpeza
    if not column.strip():

        column = "unknown_column"

    if column != original.lower():

        NORMALIZATION_STATS["renamed_columns"] += 1

    return column



def normalize_dataframe_columns(dataframe):

    used_columns = set()

    normalized_columns = []

    for column in dataframe.columns:

        clean = normalize_column_name(
            column
        )

        original = clean

        counter = 1

        while clean in used_columns:

            NORMALIZATION_STATS[
                "duplicate_columns"
            ] += 1

            clean = (
                f"{original}_{counter}"
            )

            counter += 1

        used_columns.add(clean)

        normalized_columns.append(
            clean
        )

    return normalized_columns


# =========================
# DATABASE
# =========================

db = SQLiteManager(
    DATABASE_PATH
)

db.connect()


# =========================
# LOAD ALL CSV FILES
# =========================

csv_files = list(
    CSV_ROOT.rglob("*.csv")
)

print(
    f"Found {len(csv_files)} CSV files"
)

# =========================
# CREATE TABLES
# =========================

for csv_file in csv_files:

    try:
        # =====================
        # LOAD CSV
        # =====================

        dataframe = pd.read_csv(

            csv_file,

            engine="python",

            on_bad_lines="skip"
        )

        # =====================
        # FILL NULLS
        # =====================

        dataframe = dataframe.fillna("")

        # =====================
        # NORMALIZE COLUMNS
        # =====================

        dataframe.columns = (
            normalize_dataframe_columns(
                dataframe
            )
        )


        invalid_columns = [

            col

            for col in dataframe.columns

            if not str(col).strip()
        ]

        if invalid_columns:

            print("\n====================")
            print("INVALID COLUMN FOUND")
            print("====================")
            print(f"File: {csv_file}")
            print(f"Columns: {list(dataframe.columns)}")
            print("====================\n")

        # =====================
        # CREATE TABLE NAME
        # =====================

        parent_folder = (
            csv_file.parent.name
        )

        folder_id = (
            parent_folder
            .replace("-csv", "")
        )

        table_id = (
            csv_file.stem
        )

        table_name = (
            f"table_{folder_id}_{table_id}"
        ).lower()

        # =====================
        # SAVE TO SQLITE
        # =====================

        db.create_table_from_dataframe(

            dataframe,

            table_name
        )

        print(
            f"Loaded {table_name}"
        )

    except Exception as e:
        print(
            f"Error processing "
            f"{csv_file}: {e}"
        )

print("\n====================")
print("NORMALIZATION REPORT")
print("====================")

for key, value in NORMALIZATION_STATS.items():

    print(
        f"{key}: {value}"
    )

stats_df = pd.DataFrame(

    [NORMALIZATION_STATS]

)

stats_df.to_csv(

    "results/schema_normalization_report.csv",

    index=False
)


# =========================
# FINISH
# =========================

db.disconnect()

print(
    "Database creation completed."
)