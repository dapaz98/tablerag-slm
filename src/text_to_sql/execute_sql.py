import sqlite3
import pandas as pd


class SQLExecutor:

    def __init__(
        self,
        database_path
    ):

        self.database_path = database_path


    def execute_query(
        self,
        query
    ):

        try:

            conn = sqlite3.connect(self.database_path)

            result = pd.read_sql_query(
                query,
                conn
            )

            conn.close()

            return result

        except Exception as e:

            print(f"SQL execution error: {e}")

            return None