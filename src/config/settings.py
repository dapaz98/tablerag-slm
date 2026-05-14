from pathlib import Path

class ProcessConfig():
    def __init__(self):
        root_dir = Path.cwd()

        self.csv_wikitq = (root_dir /"data/raw/WikiTableQuestions/csv")

        self.database_path = (root_dir /"data/databases/wikitablequestions.db")