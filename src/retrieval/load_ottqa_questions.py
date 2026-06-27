import json


class OTTQALoader:

    def __init__(self, dataset_path):

        self.dataset_path = dataset_path

        self.questions = []


    def load_dataset(self):

        with open(
            self.dataset_path,
            "r",
            encoding="utf-8"
        ) as f:

            self.questions = json.load(f)


    def get_all_questions(self):

        return self.questions