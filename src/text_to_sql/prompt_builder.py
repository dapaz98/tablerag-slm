from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class PromptBuilder:

    def __init__(self):

        # modelo leve para schema linking
        # carrega só quando necessário
        self._schema_model = None


    # ==========================================
    # SCHEMA LINKING
    # ==========================================

    def _get_schema_model(self):

        if self._schema_model is None:

            self._schema_model = SentenceTransformer(
                "BAAI/bge-small-en-v1.5"
            )

        return self._schema_model


    def _select_relevant_columns(
        self,
        question,
        columns,
        top_k=5
    ):
        """
        Selects the most relevant columns
        for the question using BGE Small.

        Always keeps at least top_k columns
        or all columns if fewer exist.
        """

        if len(columns) <= top_k:
            return columns

        model = self._get_schema_model()

        question_emb = model.encode(
            [question]
        )

        column_embs = model.encode(columns)

        scores = cosine_similarity(
            question_emb,
            column_embs
        )[0]

        top_indices = np.argsort(scores)[::-1][:top_k]

        # preserva a ordem original das colunas
        top_indices_sorted = sorted(top_indices)

        return [
            columns[i]
            for i in top_indices_sorted
        ]


    # ==========================================
    # BASE INSTRUCTIONS
    # ==========================================

    def _base_instructions(self):

        return """
        You are an expert Text-to-SQL assistant.

        Your task is to generate a syntactically correct SQLite query based ONLY on the provided schema and question.

        Rules:
        - Return ONLY the SQL query.
        - Do NOT include explanations, markdown, comments, or extra text.
        - Use ONLY tables and columns present in the schema.
        - Do NOT invent table names or column names.
        - All table and column names are normalized and must be used exactly as provided.
        - Preserve underscores (_) in table and column names.
        - If filtering text values, use the values exactly as they appear in the table content.
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
        You are an expert Text-to-SQL assistant
        Your task is to generate a syntactically correct SQLite SQL query based ONLY on the provided schema and question
        Rules:
        - Return ONLY the SQL query.
        - Do NOT include explanations, markdown, comments, or extra text.
        - Use ONLY tables and columns present in the schema.
        - Do NOT invent column names or table names.

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
    # CONTENT-AWARE
    # ==========================================

    def build_content_aware_prompt(
        self,
        question,
        table_name,
        columns,
        sample_rows
    ):

        schema = ", ".join(columns)

        rows_text = sample_rows.to_string(
            index=False
        )

        prompt = f"""
        {self._base_instructions()}

        Database schema:

        Table:
        {table_name}

        Columns:
        {schema}

        Sample rows:

        {rows_text}

        User question:
        {question}

        SQL:
        """

        return prompt.strip()


    # ==========================================
    # EMBEDDING-RETRIEVAL
    # ==========================================

    def build_embedding_prompt(
        self,
        question,
        table_name,
        columns,
        top_rows
    ):

        schema = ", ".join(columns)

        retrieved_rows = "\n".join(top_rows)

        prompt = f"""
        You are an expert Text-to-SQL assistant.

        Table:
        {table_name}

        Columns:
        {schema}

        Relevant rows:
        {retrieved_rows}

        Question:
        {question}

        Generate ONLY SQLite SQL.
        """

        return prompt.strip()


    # ==========================================
    # EMBEDDING-CONTENT-AWARE
    # ==========================================

    def build_embedding_content_aware_prompt(
        self,
        question,
        table_name,
        columns,
        retrieved_rows
    ):

        schema = ", ".join(columns)

        rows_text = "\n".join(retrieved_rows)

        prompt = f"""
        You are an expert Text-to-SQL assistant.

        Generate ONLY valid SQLite SQL.

        Table:
        {table_name}

        Columns:
        {schema}

        Relevant rows retrieved from the table:

        {rows_text}

        Important:
        - Use only the provided schema.
        - Pay attention to values shown in the retrieved rows.
        - Return only SQL.
        - Do not explain anything.

        Question:
        {question}

        SQL:
        """

        return prompt.strip()


    # ==========================================
    # EMBEDDING-RETRIEVAL + SCHEMA LINKING
    # ==========================================

    def build_schema_linking_prompt(
        self,
        question,
        table_name,
        columns,
        top_rows,
        top_k_columns=5
    ):
        """
        Combines embedding retrieval with
        schema linking — selects only the
        most relevant columns for the question
        before building the prompt.

        This reduces hallucinated column names
        by narrowing the schema context.
        """

        # schema linking — filtra colunas relevantes
        relevant_columns = (
            self._select_relevant_columns(
                question,
                columns,
                top_k=top_k_columns
            )
        )

        schema = ", ".join(relevant_columns)

        retrieved_rows = "\n".join(top_rows)

        prompt = f"""
        You are an expert Text-to-SQL assistant.

        Generate ONLY valid SQLite SQL.

        Table:
        {table_name}

        Most relevant columns for this question:
        {schema}

        Relevant rows retrieved from the table:

        {retrieved_rows}

        Important:
        - Use ONLY the columns listed above.
        - Do NOT invent column names.
        - Return only SQL, no explanation.

        Question:
        {question}

        SQL:
        """

        return prompt.strip()