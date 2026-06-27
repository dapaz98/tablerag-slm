from sentence_transformers import CrossEncoder


class TableReranker:
    """
    Re-ranks retrieved tables using a
    lightweight cross-encoder model.

    Runs fully local — no API dependency.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    """

    def __init__(
        self,
        model_name=(
            "cross-encoder/"
            "ms-marco-MiniLM-L-6-v2"
        )
    ):

        self.model = CrossEncoder(
            model_name
        )


    def rerank(
        self,
        question,
        candidate_ids,
        candidate_texts,
        top_k=10
    ):
        """
        Re-ranks candidates by relevance
        to the question.

        Args:
            question       : user question string
            candidate_ids  : list of table IDs
            candidate_texts: list of table texts
            top_k          : number of results to return

        Returns:
            list of table IDs reranked by score
        """

        if not candidate_ids:
            return []

        # build (question, table_text) pairs
        pairs = [
            (question, text)
            for text in candidate_texts
        ]

        # score each pair
        scores = self.model.predict(pairs)

        # sort by score descending
        ranked = sorted(
            zip(candidate_ids, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # return top_k ids
        return [
            table_id
            for table_id, _
            in ranked[:top_k]
        ]