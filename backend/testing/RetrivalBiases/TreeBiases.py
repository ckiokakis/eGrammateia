import json
import numpy as np
import pandas as pd
from itertools import product
from typing import List, Dict, Optional
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load Evaluation Set
with open("questions_tree.json", "r", encoding="utf-8") as f:
    eval_set = json.load(f)

# Load chunks to access their index-based IDs
with open("chunks_tree.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
    chunk_id_map = {chunk["data"].strip(): idx for idx, chunk in enumerate(chunks)}

class TreeRAGPipeline:
    def __init__(self, index_path: str, fallback_model: str, chunks: List[Dict]):
        self.embeddings = HuggingFaceEmbeddings(model_name=fallback_model)
        self.store = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
        self.llm = None
        self.chunk_map: Dict[str, int] = {chunk["data"].strip(): idx for idx, chunk in enumerate(chunks)}

    def retrieve(self, query: str, raw_bias, path_bias, context_bias, top_k: int = 5) -> List[Dict]:
        docs = self.store.similarity_search(query, k=top_k)
        raw_scores = np.array([1.0] * len(docs), dtype=np.float32)
        min_score, max_score = raw_scores.min(), raw_scores.max()
        if max_score - min_score > 1e-6:
            raw_scores = (raw_scores - min_score) / (max_score - min_score)
        else:
            raw_scores = np.ones_like(raw_scores) * 0.5  # Flat scores if all the same
        query_emb = np.array(self.embeddings.embed_query(query), dtype=np.float32)
        path_texts = [" > ".join(doc.metadata.get("path", ["<unknown>"])) for doc in docs]
        content_texts = [doc.page_content for doc in docs]
        path_embs = np.array(self.embeddings.embed_documents(path_texts), dtype=np.float32)
        content_embs = np.array(self.embeddings.embed_documents(content_texts), dtype=np.float32)

        def cos_sim(a, B):
            return (B @ a) / (np.linalg.norm(a) * np.linalg.norm(B, axis=1) + 1e-8)

        def softmax(x):
            ex = np.exp(x - x.max())
            return ex / ex.sum()

        prob_combined = softmax(
            raw_bias * softmax(raw_scores) +
            path_bias * softmax(cos_sim(query_emb, path_embs)) +
            context_bias * softmax(cos_sim(query_emb, content_embs))
        )

        scored = sorted(zip(docs, prob_combined), key=lambda x: x[1], reverse=True)

        results = []
        cum = 0.0
        for doc, score in scored:
            if score < 0.005 or cum >= 0.75:
                break
            cum += score
            doc_id = self.chunk_map.get(doc.page_content.strip(), -1)
            results.append({"id": doc_id, "content": doc.page_content, "score": score})

        return results

# Bias combinations
bias_steps = np.arange(0.0, 1.01, 0.1)
bias_combinations = [
    (r, p, c) for r, p, c in product(bias_steps, repeat=3) if abs(r + p + c - 1.0) < 1e-5
]

# bias_combinations = [
#     (0.6, 0.2, 0.2),
#     (0.7, 0.15, 0.15),
#     (0.5, 0.1, 0.4),
#     (0.4, 0.4, 0.2),
#     (0.33, 0.33, 0.34)
# ]

# Initialize pipeline
pipeline = TreeRAGPipeline(
    index_path="faiss_index_tree",
    fallback_model="lighteternal/stsb-xlm-r-greek-transfer",
    chunks=chunks
)

def precision_at_k(retrieved: List[int], relevant: List[int], k: int = 5) -> float:
    return sum(1 for r in retrieved[:k] if r in relevant) / k

def recall_at_k(retrieved: List[int], relevant: List[int], k: int = 5) -> float:
    if not relevant:
        return 0.0
    return sum(1 for r in retrieved[:k] if r in relevant) / len(relevant)

def success_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    return 1.0 if any(r in relevant for r in retrieved[:k]) else 0.0

results = []
for raw_b, path_b, ctx_b in tqdm(bias_combinations, desc="Evaluating bias combinations"):
    precs, recalls, succ1s, succ3s = [], [], [], []
    skipped = 0

    for item in eval_set:
        query = item["query"]
        relevant_ids = item["relevant_ids"]
        retrieved_docs = pipeline.retrieve(query, raw_bias=raw_b, path_bias=path_b, context_bias=ctx_b)
        retrieved_ids = [doc["id"] for doc in retrieved_docs if doc["id"] != -1]

        if not retrieved_ids:
            skipped += 1
            continue

        precs.append(precision_at_k(retrieved_ids, relevant_ids, k=5))
        recalls.append(recall_at_k(retrieved_ids, relevant_ids, k=5))
        succ1s.append(success_at_k(retrieved_ids, relevant_ids, k=1))
        succ3s.append(success_at_k(retrieved_ids, relevant_ids, k=3))

    results.append({
        "raw_bias": raw_b,
        "path_bias": path_b,
        "context_bias": ctx_b,
        "MP@5": round(np.mean(precs) if precs else 0.0, 4),
        "MR@5": round(np.mean(recalls) if recalls else 0.0),
        "Success@1": round(np.mean(succ1s) if succ1s else 0.0),
        "Success@3": round(np.mean(succ3s) if succ3s else 0.0),
        "Skipped": skipped
    })

df = pd.DataFrame(results).sort_values("MP@5", ascending=False)
df.to_csv("bias_tuning_detailed.csv", index=False)

print("\nTop 5 bias combinations (by MP@5):")
print(df.head())