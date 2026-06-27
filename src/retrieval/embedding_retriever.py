from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class EmbeddingRetriever:

    def __init__(
        self,
        model_name="BAAI/bge-small-en-v1.5"
    ):

        self.model = SentenceTransformer(
            model_name
        )


    def retrieve_top_k_rows(
        self,
        question,
        rows,
        k=5
    ):
        """
        rows = list[str]
        """

        if len(rows) == 0:

            return []


        question_embedding = (
            self.model.encode(
                [question]
            )
        )

        row_embeddings = (
            self.model.encode(
                rows
            )
        )


        similarities = cosine_similarity(
            question_embedding,
            row_embeddings
        )[0]


        top_indices = np.argsort(
            similarities
        )[::-1][:k]


        top_rows = [

            rows[idx]

            for idx in top_indices
        ]

        return top_rows
    

    def build_index(
        self,
        table_ids,
        table_texts
    ):

        self.table_ids = table_ids

        self.table_embeddings = (
            self.model.encode(
                table_texts,
                show_progress_bar=True
            )
        )


    def retrieve(
        self,
        question,
        top_k=10
    ):

        question_embedding = (
            self.model.encode(
                [question]
            )
        )

        similarities = (
            cosine_similarity(
                question_embedding,
                self.table_embeddings
            )[0]
        )

        top_indices = np.argsort(
            similarities
        )[::-1][:top_k]

        return [

            self.table_ids[idx]

            for idx
            in top_indices
        ]