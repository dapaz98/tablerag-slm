import sqlite3


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