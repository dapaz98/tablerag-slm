class OTTQAMetrics:

    @staticmethod
    def hits_at_k(
        gold_table,
        retrieved_tables,
        k
    ):

        return int(
            gold_table
            in retrieved_tables[:k]
        )


    @staticmethod
    def reciprocal_rank(
        gold_table,
        retrieved_tables
    ):

        try:

            rank = (
                retrieved_tables.index(
                    gold_table
                )
                + 1
            )

            return 1 / rank

        except ValueError:

            return 0.0