from together import Together


class TogetherSQLGenerator:

    def __init__(self, api_key: str, model_name: str):
        """
        Inicializa o gerador de SQL dinamicamente com qualquer modelo da Together AI.
        """
        self.client = Together(api_key=api_key)
        self.model_name = model_name
        
    def generate_sql(
        self,
        prompt,
        retries=5
    ):

        for attempt in range(retries):

            try:

                response = (
                    self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=0,
                        max_tokens=256
                    )
                )

                return (
                    response
                    .choices[0]
                    .message
                    .content
                    .strip()
                )

            except Exception as e:

                error_msg = str(e).lower()

                is_rate_limit = any([
                    "rate limit" in error_msg,
                    "429" in error_msg,
                    "too many requests" in error_msg,
                ])

                is_timeout = any([
                    "timeout" in error_msg,
                    "timed out" in error_msg,
                ])

                if is_rate_limit:

                    wait = 30 * (attempt + 1)

                    print(
                        f"[RATE LIMIT] "
                        f"Attempt {attempt+1}/{retries}. "
                        f"Waiting {wait}s..."
                    )


                elif is_timeout:

                    wait = 10 * (attempt + 1)

                    print(
                        f"[TIMEOUT] "
                        f"Attempt {attempt+1}/{retries}. "
                        f"Waiting {wait}s..."
                    )


                else:
                    raise e

        raise Exception(
            f"Together exhausted {retries} retries."
        )