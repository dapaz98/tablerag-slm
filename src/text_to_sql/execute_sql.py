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

        conn = sqlite3.connect(
            self.database_path
        )

        try:

            result = pd.read_sql_query(
                query,
                conn
            )

            return result

        finally:

            conn.close()