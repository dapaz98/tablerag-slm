import pandas as pd
from pathlib import Path
from src.config.settings import ProcessConfig
from src.preprocessing.database import SQLiteManager

config = ProcessConfig()

# =========================
# CONFIG
# =========================

CSV_ROOT = Path(config.csv_wikitq)

DATABASE_PATH = config.database_path


# =========================
# DATABASE
# =========================

db = SQLiteManager(DATABASE_PATH)

db.connect()


# =========================
# LOAD ALL CSV FILES
# =========================

csv_files = list(CSV_ROOT.rglob("*.csv"))

print(f"Found {len(csv_files)} CSV files")


# =========================
# CREATE TABLES
# =========================
for csv_file in csv_files:

    try:

        # LOAD CSV
        dataframe = pd.read_csv(
            csv_file,
            engine="python",
            on_bad_lines="skip"
        )

        # FILL NULL VALUES
        dataframe = dataframe.fillna("")

        # NORMALIZE COLUMN NAMES
        dataframe.columns = [

            str(col)
                .strip()
                .replace(" ", "_")
                .replace("-", "_")
                .replace("%", "percent")
                .replace("/", "_")
                .replace("(", "")
                .replace(")", "")

            for col in dataframe.columns
        ]

        # CREATE TABLE NAME
        # Example:
        # 772.csv -> table_772

        # table_id = csv_file.stem

        # table_name = f"table_{table_id}"
        parent_folder = csv_file.parent.name

        folder_id = parent_folder.replace("-csv", "")

        table_id = csv_file.stem

        table_name = f"table_{folder_id}_{table_id}"

        # LOAD INTO SQLITE
        db.create_table_from_dataframe(
            dataframe,
            table_name
        )

        print(f"Loaded {table_name}")

    except Exception as e:

        print(f"Error processing {csv_file}: {e}")


# =========================
# FINISH
# =========================

db.disconnect()

print("Database creation completed.")