#!/usr/bin/env python3
import json
import time
from pathlib import Path

from tabulate import tabulate
from pytrec_eval import RelevanceEvaluator

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ─── Utility Functions ────────────────────────────────────────────────────
def load_chunks(path: str):
    """Load chunk list from JSON file (list of {"id":..., "data":...})."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts, ids = [], []
    for item in data:
        texts.append(item['data'])
        ids.append(str(item.get('id', len(ids))))
    return texts, ids


def load_json(path: str):
    """Generic JSON loader for queries or qrels."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ─── Core Evaluation ─────────────────────────────────────────────────────
def evaluate_model(chunks, chunk_ids, queries, qrels, model_name, top_k=5):
    """
    Build FAISS index for chunks with the given embedding model,
    retrieve top_k for each query, and compute IR metrics.
    """
    # 1. Embed & index
    emb = HuggingFaceEmbeddings(model_name=model_name)
    t0 = time.time()
    vs = FAISS.from_texts(chunks, emb, metadatas=[{"id": cid} for cid in chunk_ids])
    embed_time = time.time() - t0

    # 2. Run retrievals
    run = {}
    for qid, qtext in queries.items():
        results = vs.similarity_search_with_score(qtext, k=top_k)
        run[qid] = {doc.metadata['id']: float(score) for doc, score in results}

    # 3. Evaluate with pytrec_eval (drop unsupported recall measure)
    metrics_set = {f"P_1", f"P_{top_k}", "map", "recip_rank", f"ndcg_cut_{top_k}"}
    evaluator = RelevanceEvaluator(qrels, metrics_set)
    per_q = evaluator.evaluate(run)

    # 4. Aggregate per-query metrics
    agg = {m: sum(per_q[qid][m] for qid in per_q) / len(per_q) for m in metrics_set}
    print(agg)
    agg.update({"model": model_name, "embed_time_s": round(embed_time, 2)})
    return agg


def main(chunks_path="../preprocessing/chunks.json", queries_path="queries.json", qrels_path="qrels.json"):
    # Load data
    chunks, chunk_ids = load_chunks(chunks_path)
    queries = load_json(queries_path)
    qrels   = load_json(qrels_path)

    models = [
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "sentence-transformers/use-cmlm-multilingual",
        "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        "sentence-transformers/distiluse-base-multilingual-cased-v2",
        "google-bert/bert-base-multilingual-cased",
        "intfloat/multilingual-e5-small",
        "google-bert/bert-base-multilingual-uncased",
        "lighteternal/stsb-xlm-r-greek-transfer",
        "nlpaueb/bert-base-greek-uncased-v1"
    ]

    results = []
    for model in models:
        print(f"Evaluating {model}...")
        try:
            res = evaluate_model(chunks, chunk_ids, queries, qrels, model, top_k=5)
        except Exception as e:
            print(f"Error with {model}: {e}")
            continue
        results.append(res)

    # Summary table
    print("\n## IR Evaluation Results ##")
    print(tabulate(results, headers="keys", tablefmt="github", floatfmt=".4f"))


if __name__ == "__main__":
    main()
