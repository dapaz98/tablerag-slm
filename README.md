# TableRAG

Research project focused on table retrieval and natural language to SQL query generation (*Text-to-SQL*), using the **WikiTableQuestions (WTQ)** and **OTT-QA** datasets.

> **⚠️ Environment notice:** This project was developed and tested on **Linux (Ubuntu)**. All setup instructions below assume a Linux environment. If you are using Windows or macOS, some steps — particularly Ollama installation, SQLite setup, and Conda paths — may require adjustments.

---

## Overview

The project currently supports two main pipelines:

- **OTT-QA pipeline** — retrieves relevant tables given a natural language question
- **WTQ pipeline** — generates SQL from a natural language question and a table (*Text-to-SQL*)

Evaluated models include local SLMs via Ollama and larger LLMs via Together AI.

---

## Environment Setup

### 1. Create the Conda virtual environment

```bash
conda env create -f environment.yml
conda activate tablerag
```

### 2. Install SQLite3 (Ubuntu)

```bash
sudo apt-get install sqlite3
```

### 3. Build the WikiTableQuestions database

The script below processes the dataset CSVs and generates the `wikitablequestions.db` file:

```bash
python -m src.preprocessing.build_wtq_database
```

> To rebuild the database from scratch, simply run the same command again.

---

## Models

### Local SLMs — Ollama

Install Ollama and pull the desired models:

```bash
curl -fsSL https://ollama.com/install.sh | sh

ollama pull llama3.2:3b
ollama pull deepseek-coder:6.7b
ollama pull qwen2.5-coder:7b
ollama pull gemma4:e4b
```

### LLMs via Together AI

Install the client and configure your API key in the `.env` file at the project root:

```bash
pip install together
```

Create a `.env` file at the project root if it doesn't exist yet:

```env
# .env
TOGETHER_API_KEY=your_api_key_here
```

> The pipeline reads this key automatically via `python-dotenv`. Without it, Together AI models will not work.

Available models via Together AI:

- `openai/gpt-oss-120b`
- `meta-llama/Llama-3.3-70B-Instruct-Turbo`

---

## Running the Pipelines

### Table retrieval pipeline (OTT-QA)

```bash
python -m src.pipeline.run_ottqa_pipeline
```

### Text-to-SQL pipeline (WTQ)

```bash
python -m src.pipeline.run_wtq_pipeline
```

### Running generated SQL queries (optional)

To manually inspect or execute the generated SQL queries against the database:

```bash
sqlite3 data/databases/wikitablequestions.db
```

Then run any SQL query directly in the SQLite shell:

```sql
SELECT * FROM some_table WHERE condition;
```

To exit:

```bash
.quit
```

---

## Results and Analysis

Analysis notebooks and generated results are available in the repository. The analyses cover retrieval metrics and SQL execution accuracy across models and prompting methods.

---

## Future Work

- Abstract the retrieval and Text-to-SQL model execution into a unified interface
- Support end-to-end execution (retrieval → SQL) for a full dataset in a single command