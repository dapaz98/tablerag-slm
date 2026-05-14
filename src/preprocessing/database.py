import sqlite3
import pandas as pd


class SQLiteManager:

    def __init__(self, db_path):

        self.db_path = db_path
        self.connection = None


    def connect(self):

        self.connection = sqlite3.connect(self.db_path)


    def disconnect(self):

        if self.connection:
            self.connection.close()
            self.connection = None


    def execute_query(self, query):

        if not self.connection:
            raise Exception("Database connection is not established.")

        try:

            cursor = self.connection.cursor()
            cursor.execute(query)

            self.connection.commit()

            return cursor.fetchall()

        except Exception as e:

            print(f"SQL execution error: {e}")
            return None


    def create_table_from_dataframe(
        self,
        dataframe,
        table_name
    ):

        try:

            dataframe.to_sql(
                table_name,
                self.connection,
                index=False,
                if_exists="replace"
            )

        except Exception as e:

            print(f"Error creating table {table_name}: {e}")