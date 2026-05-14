class PromptBuilder:


    # ==========================================
    # BASE INSTRUCTIONS
    # ==========================================

    def _base_instructions(self):

        return """
                    You are an expert Text-to-SQL assistant.

                    Your task is to generate a syntactically correct SQLite SQL query based ONLY on the provided schema and question.

                    Rules:
                    - Return ONLY the SQL query.
                    - Do NOT include explanations, markdown, comments, or extra text.
                    - Use ONLY tables and columns present in the schema.
                    - Do NOT invent column names or table names.
                    - If the question cannot be answered using the schema, return exactly:
                    CANNOT_ANSWER
                    - Generate valid SQLite syntax only.
                    - Prefer explicit column names instead of SELECT *.
                    - Use table aliases when helpful.
                    - If aggregation is needed, use proper GROUP BY clauses.
                    - If filtering text, use LIKE for partial matches when appropriate.
                    - Handle NULL values safely when relevant.
                    - Use LIMIT when the question asks for top/best/highest/lowest results.
                    - For date comparisons, use SQLite-compatible date functions only.
                """


    # ==========================================
    # ZERO-SHOT
    # ==========================================

    def build_zero_shot_prompt(
        self,
        question,
        table_name,
        columns
    ):

            schema = ", ".join(columns) 
            prompt = f"""
            {self._base_instructions()}

            Database schema:

            Table: {table_name}

            Columns:
            {schema}

            User question:
            {question}

            SQL:
            """

            return prompt.strip()


    # ==========================================
    # FEW-SHOT
    # ==========================================

    def build_few_shot_prompt(
        self,
        question,
        table_name,
        columns
    ):

        schema = ", ".join(columns)

        prompt = f"""
{self._base_instructions()}

========================================
Example 1

Table: table_example_1

Columns:
Player, Goals, Team

Question:
Which player scored the most goals?

SQL:
SELECT Player
FROM table_example_1
ORDER BY Goals DESC
LIMIT 1;

========================================
Example 2

Table: table_example_2

Columns:
Country, Population

Question:
Which country has population greater than 1000000?

SQL:
SELECT Country
FROM table_example_2
WHERE Population > 1000000;

========================================
Example 3

Table: table_example_3

Columns:
Year, Winner

Question:
Who won in 2010?

SQL:
SELECT Winner
FROM table_example_3
WHERE Year = 2010;

========================================

Now solve the following task.

Database schema:

Table: {table_name}

Columns:
{schema}

User question:
{question}

SQL:
"""

        return prompt.strip()


    # ==========================================
    # CHAIN-OF-THOUGHT
    # ==========================================

    def build_cot_prompt(
        self,
        question,
        table_name,
        columns
    ):

        schema = ", ".join(columns)

        prompt = f"""
{self._base_instructions()}

Think step-by-step about:
- which columns are relevant,
- what filtering conditions are needed,
- whether ordering or aggregation is required.

Then generate the final SQL query.

Return ONLY the SQL query.

Database schema:

Table: {table_name}

Columns:
{schema}

User question:
{question}

SQL:
"""

        return prompt.strip()