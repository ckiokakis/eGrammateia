import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    precision_score, recall_score, f1_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from IsoScore.IsoScore import IsoScore
from collections import defaultdict

# Retrieval metrics helpers
def precision_at_k(relevant, retrieved, k):
    return len(set(retrieved[:k]) & set(relevant)) / k

def average_precision(relevant, retrieved):
    hits, score = 0, 0.0
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / len(relevant) if relevant else 0.0

def mean_reciprocal_rank(relevant, retrieved):
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0

def dcg_at_k(relevances, k):
    return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevances[:k]))

def ndcg_at_k(relevant, retrieved, k):
    rels = [1 if doc in relevant else 0 for doc in retrieved]
    ideal = sorted(rels, reverse=True)
    return dcg_at_k(rels, k) / dcg_at_k(ideal, k) if dcg_at_k(ideal, k) > 0 else 0.0

# Load chunks
with open("chunks_tree.json", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [c["data"] for c in chunks]
labels = [" / ".join(c["path"]) for c in chunks]  # use path as label for classification

# Precompute TF-IDF distances for baseline distance preservation
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
tfidf_emb = tfidf.fit_transform(texts)
tfidf_sim = cosine_similarity(tfidf_emb)
tfidf_dist = 1 - tfidf_sim  # cosine distance

results = defaultdict(dict)

for model_name in [
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
]:
    # 1. Encode
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    # 2. Classification (k-NN)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        embs, labels, range(len(texts)), test_size=0.2, stratify=labels, random_state=42
    )
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro")
    results[model_name]["accuracy"] = acc
    results[model_name]["precision"] = prec
    results[model_name]["recall"] = rec
    results[model_name]["f1"] = f1

    # 3. Retrieval (treat each test item as a query)
    p_at_5, ap_sum, mrr_sum, ndcg_sum = 0, 0, 0, 0
    for qi in idx_test:
        # relevant = all chunks with same label
        relevant = [i for i, lab in enumerate(labels) if lab == labels[qi] and i != qi]
        sims = list(enumerate(cosine_similarity([embs[qi]], embs)[0]))
        sims.sort(key=lambda x: x[1], reverse=True)
        retrieved = [i for i, _ in sims if i != qi]
        p_at_5 += precision_at_k(relevant, retrieved, 5)
        ap_sum += average_precision(relevant, retrieved)
        mrr_sum += mean_reciprocal_rank(relevant, retrieved)
        ndcg_sum += ndcg_at_k(relevant, retrieved, 5)
    n_queries = len(idx_test)
    results[model_name]["P@5"] = p_at_5 / n_queries
    results[model_name]["MAP"] = ap_sum / n_queries
    results[model_name]["MRR"] = mrr_sum / n_queries
    results[model_name]["nDCG@5"] = ndcg_sum / n_queries

    # 4. IsoScore
    iso = IsoScore(embs)
    results[model_name]["isotropy"] = iso

    # 5. Nearest Neighbor Overlap (N2O)
    # compute overlap between top-10 neighbors sets for each point
    sims_full = cosine_similarity(embs)
    n2o_sum = 0
    for i in range(len(embs)):
        nn_orig = set(np.argsort(-sims_full[i])[:10])
        # for a second embedding space you would compare two spaces—here approximate by self-overlap
        n2o_sum += len(nn_orig & nn_orig) / 10
    results[model_name]["N2O"] = n2o_sum / len(embs)

    # 6. Spearman correlation between TF-IDF and embedding distances
    emb_dist = 1 - cosine_similarity(embs)
    rho, _ = spearmanr(tfidf_dist.flatten(), emb_dist.flatten())
    results[model_name]["spearman"] = rho

# Display results
import pandas as pd
df = pd.DataFrame(results).T
print(df.sort_values("f1", ascending=False))
