import time
from pathlib import Path
import pandas as pd

from src.retrieval.load_ottqa_questions import OTTQALoader
from src.retrieval.load_ottqa_tables import OTTQATableLoader
from src.retrieval.embedding_retriever import EmbeddingRetriever
from src.retrieval.reranker import TableReranker
from src.evaluation.ottqa_metrics import OTTQAMetrics

# =====================================
# CONFIG
# =====================================
MAX_SAMPLES = 2214
RETRIEVAL_TOP_K = 20   
RERANK_TOP_K    = 10   
USE_RERANKER = False    

QUESTIONS_PATH = "data/raw/OTT-QA/released_data/dev.json"
TABLES_PATH = "data/raw/OTT-QA/traindev_tables.json"
RESULTS_PATH = "results/ottqa_results_reranked.csv" if USE_RERANKER else "results/ottqa_results.csv"

# =====================================
# LOAD DATA & BUILD INDEX
# =====================================
question_loader = OTTQALoader(QUESTIONS_PATH)
question_loader.load_dataset()
questions = question_loader.get_all_questions()

table_loader = OTTQATableLoader(TABLES_PATH)
table_loader.load_tables()
tables = table_loader.get_all_tables()

table_ids, table_texts = [], []
for table_id, table in tables.items():
    table_text = table_loader.table_to_text(table)
    table_ids.append(table_id)
    table_texts.append(table_text)

table_text_map = dict(zip(table_ids, table_texts))
print(f"Tables indexed: {len(table_ids)}")

retriever = EmbeddingRetriever()
retriever.build_index(table_ids, table_texts)

if USE_RERANKER:
    print("Loading re-ranker...")
    reranker = TableReranker()
    print("Re-ranker loaded.")

# =====================================
# RUN RETRIEVAL WITH ADVANCED LOGGING (O código novo entra AQUI)
# =====================================
results = []
print(f"\nIniciando busca para {MAX_SAMPLES} samples (reranker={USE_RERANKER})...")

start_retrieval_loop = time.time()

for sample in questions[:MAX_SAMPLES]:
    question_id = sample["question_id"]
    question    = sample["question"]
    gold_table  = sample["table_id"]

    # --- STEP 1: EMBEDDING RETRIEVAL (BI-ENCODER) ---
    t_start_emb = time.time()
    retrieved_tables = retriever.retrieve(question, top_k=RETRIEVAL_TOP_K)
    emb_latency_ms = (time.time() - t_start_emb) * 1000

    # Verifica se o gold estava no retrieval inicial (Top 20)
    in_initial_retrieval = gold_table in retrieved_tables
    initial_rank = retrieved_tables.index(gold_table) + 1 if in_initial_retrieval else -1

    rerank_latency_ms = 0.0
    final_rank = -1

    # --- STEP 2: RE-RANKING (CROSS-ENCODER) ---
    if USE_RERANKER:
        candidate_texts = [table_text_map.get(t, "") for t in retrieved_tables]
        
        t_start_rerank = time.time()
        final_tables = reranker.rerank(
            question,
            retrieved_tables,
            candidate_texts,
            top_k=RERANK_TOP_K
        )
        rerank_latency_ms = (time.time() - t_start_rerank) * 1000
    else:
        final_tables = retrieved_tables[:RERANK_TOP_K]

    # Rank final após ordenação (Top 10)
    in_final_results = gold_table in final_tables
    if in_final_results:
        final_rank = final_tables.index(gold_table) + 1

    # --- STEP 3: METRICS ---
    hit1 = OTTQAMetrics.hits_at_k(gold_table, final_tables, 1)
    hit5 = OTTQAMetrics.hits_at_k(gold_table, final_tables, 5)
    hit10 = OTTQAMetrics.hits_at_k(gold_table, final_tables, 10)
    mrr = OTTQAMetrics.reciprocal_rank(gold_table, final_tables)

    # --- CLASSIFICAÇÃO QUALITATIVA DE ERRO ---
    error_type = "SUCCESS"
    if not in_final_results:
        if not in_initial_retrieval:
            error_type = "BI_ENCODER_FAILURE"  # O primeiro modelo nem achou
        else:
            error_type = "RERANKER_DROP"       # O re-ranker eliminou do top 10

    results.append({
        "question_id": question_id,
        "question": question,
        "gold_table": gold_table,
        "top1_predicted": final_tables[0] if final_tables else None,
        "all_retrieved_top10": final_tables,
        "hit@1": hit1,
        "hit@5": hit5,
        "hit@10": hit10,
        "mrr": mrr,
        "initial_rank": initial_rank,
        "final_rank": final_rank,
        "error_category": error_type,
        "emb_latency_ms": emb_latency_ms,
        "rerank_latency_ms": rerank_latency_ms,
        "total_latency_ms": emb_latency_ms + rerank_latency_ms,
        "reranker_used": USE_RERANKER,
    })

total_time_seconds = time.time() - start_retrieval_loop
avg_latency_ms = (total_time_seconds / MAX_SAMPLES) * 1000

# =====================================
# SAVE CSV (Salva o arquivo rico em dados)
# =====================================
results_df = pd.DataFrame(results)
Path("results").mkdir(exist_ok=True)
results_df.to_csv(RESULTS_PATH, index=False)

# =====================================
# SUMMARY (O print final continua funcionando)
# =====================================
print("\n====================")
print("FINAL RESULTS")
print("====================")
print(f"Samples      : {len(results_df)}")
print(f"Reranker     : {USE_RERANKER}")
print(f"Hits@1       : {results_df['hit@1'].mean():.4f}")
print(f"Hits@5       : {results_df['hit@5'].mean():.4f}")
print(f"Hits@10      : {results_df['hit@10'].mean():.4f}")
print(f"MRR          : {results_df['mrr'].mean():.4f}")
print(f"Total time   : {total_time_seconds:.2f}s")
print(f"Avg latency  : {avg_latency_ms:.2f}ms")
print(f"CSV          : {RESULTS_PATH}")