import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Dict
import warnings

import requests
import websockets
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import faiss

# Transformer imports
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- Configuration ---
@dataclass(frozen=True)
class Config:
    # LLM API settings
    llm_api_url: str = os.getenv("LLM_API_URL", "http://10.240.138.254:11434/api/chat")
    llm_model: str = os.getenv("LLM_MODEL", "chat:llama3.2")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "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")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # FAISS & embeddings
    faiss_index_path: str = os.getenv("FAISS_INDEX_PATH", "./faiss_index")
    hf_embedding_model: str = os.getenv(
        "HF_EMBEDDING_MODEL",
        "lighteternal/stsb-xlm-r-greek-transfer",
    )

    # Path to JSON tree chunks (for TreeRAG)
    chunks_json_path: str = os.getenv("CHUNKS_JSON_PATH", "preprocessing/chunks.json")

    # Which retriever to use: 'flat' for VectorStoreManager, 'tree' for hierarchical
    rag_type: str = os.getenv("RAG_TYPE", "tree") 

    # WebSocket settings
    host: str = os.getenv("WS_HOST", "0.0.0.0")
    port: int = int(os.getenv("WS_PORT", "8090"))

cfg = Config()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# --- LLM Clients ---
class LLMClient:
    """HTTP-based LLM client for 'opensource' engine."""
    def __init__(self, api_url: str, model: str) -> None:
        self.api_url = api_url
        self.model = model
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def chat(self, messages: List[dict], seed: int = 42,
             temperature: float = 0.7, stream: bool = False) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {"seed": seed, "temperature": temperature},
            "stream": stream,
        }
        try:
            resp = self.session.post(self.api_url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "No response")
        except requests.RequestException as err:
            logger.error("LLM error: %s", err)
            return f"Error: {err}"

class GroqClient:
    """Wrapper for GROQ-based LLM ('groq' engine)."""
    def __init__(self, api_key: str, model: str) -> None:
        self.client = Groq(api_key=api_key)
        self.model = model

    def complete(self, prompt: str) -> str:
        comp = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
        )
        return comp.choices[0].message.content.strip()

# --- Retrieval ---
class VectorStoreManager:
    """Flat FAISS-backed store."""
    def __init__(self, index_path: str, fallback_model: str) -> None:
        idx = Path(index_path)
        meta_path = idx / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            model_name = meta.get("model_name", fallback_model)
            logger.info("Loaded embedding model: %s", model_name)
        else:
            model_name = fallback_model
            logger.warning("No metadata.json found, fallback to: %s", fallback_model)
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.store = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)

    def search(self, query: str, k: int = 5) -> List[str]:
        docs = self.store.similarity_search(query, k=k)
        return [getattr(d, "page_content", str(d)) for d in docs]

class TreeRAGPipeline:
    """
    RAG pipeline that embeds hierarchically structured chunks,
    stores them in a FAISS index, retrieves relevant context,
    and generates answers using an external LLM client.
    """

    def __init__(
        self,
        chunks_json_path: str,
        llm: 'LLMClient',
        embedding_model_name: str = 'lighteternal/stsb-xlm-r-greek-transfer',
    ):
        # External LLM client for generation
        self.llm = llm

        # 1) Embedder
        self.embedder = SentenceTransformer(embedding_model_name)

        # 2) Load chunks
        with open(chunks_json_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)

        # 3) Placeholder for FAISS index & metadata
        self.index: faiss.Index = None  # will be set in build_index()
        self.metadata: List[Dict] = []

    def build_index(self) -> None:
        """Combine path and content, embed, normalize, and build FAISS index."""
        texts = []
        meta = []
        for node in self.chunks:
            path = node['path']
            content = node['data']
            full_text = f"{' > '.join(path)}: {content}"
            texts.append(full_text)
            meta.append({'path': path, 'content': content})
        self.metadata = meta

        # embed all nodes
        embs = self.embedder.encode(texts, convert_to_tensor=False, show_progress_bar=True)
        embs = np.vstack(embs).astype('float32')
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)

        dim = embs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embs)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top_k nodes by inner product similarity."""
        if self.index is None:
            logger.info("FAISS index not found—building it now.")
            self.build_index()

        # embed + normalize query
        q_emb = self.embedder.encode([query], convert_to_tensor=False)
        q_emb = np.array(q_emb, dtype='float32')
        q_emb /= np.linalg.norm(q_emb, axis=1, keepdims=True)

        D, I = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            m = self.metadata[idx]
            results.append({
                'score': float(score),
                'path': m['path'],
                'content': m['content']
            })
        return results


# --- QA Service ---
EngineType = Literal['groq', 'opensource']

class QAService2:
    """Retrieval + generation orchestrator."""

    def __init__(
        self,
        groq: 'GroqClient',
        llm: 'LLMClient',
        vs: 'VectorStoreManager',
        tree: TreeRAGPipeline,
        use_tree: bool,
        chunks_json_path: Optional[str] = None
    ) -> None:
        self.groq = groq
        self.llm = llm
        self.vs = vs
        self.tree = tree
        self.use_tree = use_tree

        # load for exact-match
        self.all_chunks = []
        if chunks_json_path:
            try:
                with open(chunks_json_path, encoding='utf-8') as f:
                    self.all_chunks = json.load(f)
            except Exception:
                logger.warning("Could not load chunks.json for exact-match.")

    def _build_prompt(self, contexts: List[str], question: str) -> str:
        info = "\n\n".join(contexts)
        return f"""
Είσαι η γραμματεία του τμήματος Ηλεκτρολόγων Μηχανικών και Τεχνολογίας Υπολογιστών του Πανεπιστημίου Πατρών. Λειτουργείς πάντα υπό τους παρακάτω κανόνες:

> **Επεξήγηση & Διευκρινίσεις**  
   - Αν η ερώτηση του χρήστη είναι ασαφής, ζήτα διευκρινίσεις.  
   - Αν δεν μπορείς να συμμορφωθείς πλήρως, εξήγησε το γιατί και πρόσφερε μερική βοήθεια αν είναι ασφαλές.
   
> **Απαντάς ΜΟΝΟ στα Ελληνικά**  
   - Χρησιμοποιείς μόνο ελληνικά, εκτός από αμετάφραστους όρους.

> **Συνεκτικός και Ορθογραφικός Έλεγχος**  
   - Διορθώνεις ορθογραφικά ή συντακτικά λάθη.

> **Μορφή Απάντησης**
   - Μην εμφανίζεις την σκέψη, τα επιχειρήματα ή τα σχόλια διόρθωσης.
   - Σύντομη και περιεκτική (1–4 προτάσεις + bullet points αν χρειάζεται).
   - Καλύπτει πλήρως την ερώτηση χωρίς επιπλέον σχόλια.

> **Απαντάς ΜΟΝΟ με βάση αυτές τις πληροφορίες:**
   {info}

Απαντάς στην ερώτηση: «{question}» ελέγχοντας και ακολουθώντας τους κανόνες. Αν υπάρχει σύγκρουση, κάνε ασφαλή ολοκλήρωση.
"""

    def generate(
        self,
        user_query: str,
        engine: EngineType = 'groq',
        k: int = 3,
        max_new_tokens: int = 256
    ) -> str:
        """
        Retrieve relevant chunks (either via tree or flat vector search),
        build a combined context block, and send to the chosen LLM backend.
        """
        # 1) Retrieve docs/chunks
        if self.use_tree:
            raw = self.tree.retrieve(user_query, k)  # List[Dict[path, content]]
            # contexts = [
            #     f"[Doc {i} | {' > '.join(d['path'])}]\n{d['content']}"
            #     for i, d in enumerate(raw, start=1)
            # ]
            contexts = [
                f"{' > '.join(d['path'])}\n{d['content']}"
                for i, d in enumerate(raw, start=1)
            ]
            # contexts = [f"\n{d['content']}" for i, d in enumerate(raw, start=1)]
        else:
            flat_chunks: List[str] = self.vs.search(user_query, k=k) or []
            # exact‐match terms in angle brackets
            terms = re.findall(r'<([^>]+)>', user_query)
            exact_matches = [
                chunk['data']
                for t in terms
                for chunk in self.all_chunks
                if t in chunk.get('data', '')
            ][: k * len(terms)]
            contexts = []
            if flat_chunks:
                contexts.append("**Ανακτημένα αποσπάσματα:**\n" + "\n\n".join(flat_chunks))
            if exact_matches:
                contexts.append("**Ακριβείς αντιστοιχίες:**\n" + "\n\n".join(exact_matches))

        # 2) Build prompt
        prompt = self._build_prompt(contexts, user_query)

        print("-"*100+f"\n{prompt}\n"+"-"*100)

        # 3) Generate
        if self.use_tree or engine != 'groq':
            # tree always uses self.llm.chat, or flat + non-groq
            return self.llm.chat([{"role": "user", "content": prompt}])
        else:
            # flat + groq engine
            return self.groq.complete(prompt, max_new_tokens=max_new_tokens)

# --- WebSocket Server ---
class WebSocketQA:
    def __init__(self, qa: QAService2, host: str, port: int) -> None:
        self.qa = qa
        self.host = host
        self.port = port

    async def _handler(self, ws: websockets.WebSocketServerProtocol) -> None:
        logger.info("Client connected: %s", ws.remote_address)
        try:
            async for raw in ws:
                data = json.loads(raw)
                query = data.get('query', '')
                engine = data.get('engine', 'groq')
                logger.info("Received query=%r engine=%s", query, engine)
                start = time.time()
                answer = self.qa.generate(query, engine=engine)
                duration = time.time() - start
                logger.info("Replied in %.2f sec", duration)
                await ws.send(answer)
                await ws.send("[END]")
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected: %s", ws.remote_address)
        except Exception:
            logger.exception("Handler error:")

    def run(self) -> None:
        logger.info("Starting server at %s:%d", self.host, self.port)
        asyncio.run(self._serve_forever())

    async def _serve_forever(self) -> None:
        async with websockets.serve(self._handler, self.host, self.port):
            await asyncio.Future()

# --- Main ---
def main() -> None:
    if not cfg.groq_api_key:
        logger.critical("Missing GROQ_API_KEY environment variable")
        return

    groq = GroqClient(api_key=cfg.groq_api_key, model=cfg.groq_model)
    llm = LLMClient(api_url=cfg.llm_api_url, model=cfg.llm_model)
    vs = VectorStoreManager(cfg.faiss_index_path, cfg.hf_embedding_model)
    tree = TreeRAGPipeline(cfg.chunks_json_path, cfg.hf_embedding_model)
    use_tree = cfg.rag_type.lower() == 'tree'
    qa = QAService2(
        groq=groq,
        llm=llm,
        vs=vs,
        tree=tree,
        use_tree=use_tree,
        chunks_json_path=cfg.chunks_json_path
    )
    server = WebSocketQA(qa, cfg.host, cfg.port)
    server.run()

if __name__ == "__main__":
    main()
