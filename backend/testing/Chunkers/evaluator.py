#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- TreeRAGPipeline (for tree mode) ---
class TreeRAGPipeline:
    """
    Hierarchical RAG pipeline with FAISS and weighted hybrid retrieval.
    """
    def __init__(
        self,
        index_path: str,
        embedding_model: str,
        chunks_json_path: Optional[str] = None,
    ):
        # --- load and index chunks for ID mapping ---
        self.content_id_map: Dict[str, int] = {}
        if chunks_json_path:
            with open(chunks_json_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                chunks_list = raw.get('data', [])
            elif isinstance(raw, list):
                chunks_list = raw
            else:
                chunks_list = []
            # map each chunk's 'data' text → its position
            for idx, chunk in enumerate(chunks_list):
                text = chunk.get('data', '').strip()
                if text:
                    self.content_id_map[text] = idx

        # Load embeddings and FAISS index
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.store = FAISS.load_local(
            index_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # 1) raw FAISS search
        docs_and_scores = self.store.similarity_search_with_relevance_scores(query, k=top_k)
        docs, raw_scores = zip(*docs_and_scores)
        raw_scores = np.array(raw_scores, dtype=np.float32)

        # 2) embed query, paths, and content
        query_emb = np.array(self.embeddings.embed_query(query), dtype=np.float32)
        # path fallback to chunk_map if metadata.path missing (like before)
        path_texts = []
        for doc in docs:
            path = doc.metadata.get("path")
            if path:
                path_texts.append(" > ".join(path))
            else:
                key = doc.page_content.strip()
                path_texts.append(" > ".join(self.content_id_map.get(key, [""])))
        content_texts = [d.page_content for d in docs]

        path_embs    = np.array(self.embeddings.embed_documents(path_texts), dtype=np.float32)
        content_embs = np.array(self.embeddings.embed_documents(content_texts), dtype=np.float32)

        # helpers
        def cos_sim(a: np.ndarray, B: np.ndarray) -> np.ndarray:
            a_norm = np.linalg.norm(a)
            B_norms = np.linalg.norm(B, axis=1)
            return (B @ a) / (a_norm * B_norms + 1e-8)

        def softmax(x: np.ndarray) -> np.ndarray:
            ex = np.exp(x - x.max())
            return ex / ex.sum()

        sim_path    = cos_sim(query_emb, path_embs)
        sim_content = cos_sim(query_emb, content_embs)

        prob_orig    = softmax(raw_scores)
        prob_path    = softmax(sim_path)
        prob_context = softmax(sim_content)

        # weighted hybrid
        w0, w1, w2 = 0.6, 0.2, 0.2
        combined_raw = w0*prob_orig + w1*prob_path + w2*prob_context
        prob_combined = softmax(combined_raw)

        # rank & package results, including our new ID lookup
        scored = sorted(zip(docs, prob_combined), key=lambda t: t[1], reverse=True)
        results: List[Dict[str, Any]] = []
        for doc, score in scored[:top_k]:
            content = doc.page_content
            results.append({
                "id":      self.content_id_map.get(content, -1),
                "content": content,
                "score":   float(score),
            })
        return results


# --- Flat RAG retriever ---
class VectorStoreManager:
    """
    Flat FAISS-backed RAG retriever. Uses chunk position as ID.
    """
    def __init__(
        self,
        index_path: str,
        embedding_model: str,
        chunks_json_path: Optional[str] = None,
    ) -> None:
        # --- load for ID mapping ---
        self.content_id_map: Dict[str, int] = {}
        if chunks_json_path:
            with open(chunks_json_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                chunks_list = raw.get('data', [])
            elif isinstance(raw, list):
                chunks_list = raw
            else:
                chunks_list = []
            for idx, chunk in enumerate(chunks_list):
                text = chunk.get('data', '').strip()
                if text:
                    self.content_id_map[text] = idx

        # embeddings & FAISS
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.store = FAISS.load_local(
            str(Path(index_path)),
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        docs = self.store.similarity_search(query, k=k)
        results: List[Dict[str, Any]] = []
        for d in docs:
            content = d.page_content
            results.append({
                "id":      self.content_id_map.get(content, -1),
                "content": content,
            })
        return results

    def search_texts(self, query: str, k: int = 5) -> List[str]:
        return [d.page_content for d in self.store.similarity_search(query, k=k)]


# --- Metrics (unchanged) ---
def precision_recall_at_k(
    retrieved: List[int],
    relevant: List[int],
    k: int
) -> Tuple[float, float]:
    retrieved_k = retrieved[:k]
    if not relevant:
        return 0.0, 0.0
    hits = len(set(retrieved_k) & set(relevant))
    return hits / k, hits / len(relevant)


def mean_average_precision(
    retrieved: List[int],
    relevant: List[int],
    k: int
) -> float:
    y_true = [1 if idx in relevant else 0 for idx in retrieved[:k]]
    y_scores = np.linspace(k, 1, num=k)
    if not any(y_true):
        return 0.0
    return average_precision_score(y_true, y_scores)


def reciprocal_rank(
    retrieved: List[int],
    relevant: List[int]
) -> float:
    for rank, idx in enumerate(retrieved, start=1):
        if idx in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_all(
    all_retrieved: List[List[int]],
    queries_meta: List[Dict[str, Any]],
    k: int
) -> Dict[str, float]:
    precisions, recalls, aps, mrrs = [], [], [], []
    for retrieved_ids, meta in zip(all_retrieved, queries_meta):
        relevant = meta.get('relevant_ids', [])
        p, r   = precision_recall_at_k(retrieved_ids, relevant, k)
        aps.append(mean_average_precision(retrieved_ids, relevant, k))
        mrrs.append(reciprocal_rank(retrieved_ids, relevant))
        precisions.append(p); recalls.append(r)
    return {
        f"Precision@{k}": np.mean(precisions),
        f"Recall@{k}":    np.mean(recalls),
        f"MAP@{k}":       np.mean(aps),
        "MRR":            np.mean(mrrs),
    }


# --- Main ---
def main(
    index_path: str,
    chunks_json: str,
    queries_json: str,
    embedding_model: str,
    k: int,
    mode: str
) -> None:
    # Load queries
    with open(queries_json, 'r', encoding='utf-8') as f:
        queries_meta = json.load(f)
    queries = [q['query'] for q in queries_meta]

    # Instantiate with chunk-position IDs
    vs = VectorStoreManager(index_path, embedding_model, chunks_json)
    tr = TreeRAGPipeline(index_path, embedding_model, chunks_json)

    all_ids: List[List[int]]   = []
    all_texts: List[List[str]] = []

    for q in queries:
        if mode == 'tree':
            items = tr.retrieve(q, top_k=k)
        else:
            items = vs.search(q, k=k)

        ids   = [item['id']      for item in items]
        texts = [item['content'] for item in items]
        all_ids.append(ids)
        all_texts.append(texts)

    # Evaluate & print
    metrics = evaluate_all(all_ids, queries_meta, k)

    for i, q in enumerate(queries):
        print(f"Query {i+1}: {q!r}")
        print(" Retrieved IDs:", all_ids[i])
        for rank, (cid, txt) in enumerate(zip(all_ids[i], all_texts[i]), start=1):
            snippet = txt.replace("\n", " ")[:100]
            print(f"   {rank}. [id={cid}] {snippet}...")
        print()

    print("=== Aggregate Metrics ===")
    for name, v in metrics.items():
        print(f"{name}: {v:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate RAG (flat or tree)")
    parser.add_argument("--index",   "-i", required=True, help="FAISS index dir")
    parser.add_argument("--chunks",  "-c", required=True, help="Chunks JSON file")
    parser.add_argument("--queries", "-q", required=True, help="Queries JSON file")
    parser.add_argument(
        "--emb_model", "-m",
        default="lighteternal/stsb-xlm-r-greek-transfer",
        help="Embedding model name"
    )
    parser.add_argument("--top_k",   "-k", type=int, default=5, help="Docs to retrieve")
    parser.add_argument(
        "--mode", "-M", choices=["flat","tree"], default="flat",
        help="Retrieval mode"
    )
    args = parser.parse_args()

    main(
        index_path=args.index,
        chunks_json=args.chunks,
        queries_json=args.queries,
        embedding_model=args.emb_model,
        k=args.top_k,
        mode=args.mode
    )
