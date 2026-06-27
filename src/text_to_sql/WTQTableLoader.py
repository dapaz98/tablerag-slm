import sqlite3
import pandas as pd


class TableContentLoader:

    def __init__(self, database_path):

        self.database_path = database_path

    def get_table_rows(
        self,
        table_name,
        limit=100
    ):

        connection = sqlite3.connect(
            self.database_path
        )

        query = f'''
        SELECT *
        FROM "{table_name}"
        LIMIT {limit}
        '''

        dataframe = pd.read_sql_query(
            query,
            connection
        )

        connection.close()

        return dataframe