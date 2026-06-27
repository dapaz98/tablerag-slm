import json


class OTTQATableLoader:

    def __init__(self, table_path):

        self.table_path = table_path

        self.tables = {}


    def load_tables(self):

        with open(
            self.table_path,
            "r",
            encoding="utf-8"
        ) as f:

            self.tables = json.load(f)


    def get_all_tables(self):

        return self.tables


    def table_to_text(self, table):

        title = table.get(
            "title",
            ""
        )

        intro = table.get(
            "intro",
            ""
        )

        headers = [

            column[0]

            for column
            in table["header"]
        ]

        schema = ", ".join(
            headers
        )

        return f"""
Title:
{title}

Schema:
{schema}

Intro:
{intro}
"""