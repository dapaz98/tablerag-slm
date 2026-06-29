from pathlib import Path
import pandas as pd


class WTQQuestionLoader:
    def __init__(self, dataset_path):

        self.dataset_path = Path(dataset_path)

        self.dataframe = None

    def load_dataset(self):

        """
        Load WikiTableQuestions TSV file.
        """

        self.dataframe = pd.read_csv(
            self.dataset_path, sep="\t", engine="python", on_bad_lines="skip"
        )

        return self.dataframe

    def get_question_by_index(self, index):

        """
        Return a single WTQ sample by dataframe index.
        """

        if self.dataframe is None:
            raise Exception("Dataset not loaded.")

        row = self.dataframe.iloc[index]

        return self._parse_row(row)

    def get_all_questions(self):

        """
        Return all questions parsed as list of dictionaries.
        """

        if self.dataframe is None:
            raise Exception("Dataset not loaded.")

        questions = []

        for _, row in self.dataframe.iterrows():

            try:

                parsed = self._parse_row(row)

                questions.append(parsed)

            except Exception as e:

                print(f"Error parsing row: {e}")

        return questions

    def _parse_row(self, row):

        """
        Parse WTQ row structure.
        """

        question = row["utterance"]

        context = row["context"]

        gold_answer = row["targetValue"]

        context_path = Path(context)

        folder_name = context_path.parent.name

        folder_id = folder_name.replace("-csv", "")

        table_id = context_path.stem

        table_name = f"table_{folder_id}_{table_id}"

        parsed = {
            "question_id": row["id"],
            "question": question,
            "table_name": table_name,
            "table_context": context,
            "gold_answer": gold_answer,
        }

        return parsed


if __name__ == "__main__":

    DATASET_PATH = "data/raw/WikiTableQuestions/" "data/random-split-1-dev.tsv"

    loader = WTQQuestionLoader(DATASET_PATH)

    dataframe = loader.load_dataset()

    print(dataframe.head())

    print("\n====================")
    print("SAMPLE QUESTION")
    print("====================\n")

    sample = loader.get_question_by_index(0)

    print(sample)
