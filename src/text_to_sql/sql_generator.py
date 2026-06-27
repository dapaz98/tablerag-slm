import ollama


class SQLGenerator:

    def __init__(
        self,
        model_name
    ):

        self.model_name = model_name


    def generate_sql(
        self,
        prompt
    ):

        print(f"\nUSING MODEL: {self.model_name}\n")

        response = ollama.chat(

        model=self.model_name,

        options={
            "temperature": 0
        },

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

        sql_query = response["message"]["content"]

        return sql_query.strip()