import sqlite3
import pandas as pd

class SchemaExtractor:

    def __init__(
        self,
        database_path
    ):

        self.database_path = database_path


    def get_table_schema(
        self,
        table_name
    ):

        conn = sqlite3.connect(self.database_path)

        cursor = conn.cursor()

        query = f"""
        PRAGMA table_info({table_name});
        """

        cursor.execute(query)

        schema_info = cursor.fetchall()

        conn.close()

        columns = [
            column[1]
            for column in schema_info
        ]

        return columns
    


    def get_sample_rows(
        self,
        table_name,
        limit=3
    ):

        connection = sqlite3.connect(
            self.database_path
        )

        query = f"""
        SELECT *
        FROM {table_name}
        LIMIT {limit}
        """

        dataframe = pd.read_sql_query(
            query,
            connection
        )

        connection.close()

        return dataframe