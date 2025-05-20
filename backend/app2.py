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
    RAG pipeline that loads a FAISS index from disk (via LangChain’s FAISS.load_local),
    uses HuggingFaceEmbeddings under the hood, and retrieves with probability confidences.

    If metadata.path is missing in indexed documents, optionally loads the original
    chunks.json to recover the `path` information based on matching content.
    """

    def __init__(
        self,
        index_path: str,
        llm: 'LLMClient',
        fallback_model: str = 'lighteternal/stsb-xlm-r-greek-transfer',
        chunks_json_path: Optional[str] = None,
    ):
        self.llm = llm

        # 0) Load chunks.json for fallback mapping (optional)
        self.chunk_map: Dict[str, List[str]] = {}
        if chunks_json_path:
            try:
                with open(chunks_json_path, encoding='utf-8') as f:
                    all_chunks = json.load(f)
                # Build mapping: content -> path
                for chunk in all_chunks:
                    content = chunk.get('data', '').strip()
                    path = chunk.get('path')
                    if content and path:
                        self.chunk_map[content] = path
                logger.info("Loaded %d chunks for path fallback", len(self.chunk_map))
            except Exception as e:
                logger.warning("Could not load chunks.json for path fallback: %s", e)

        # 1) Figure out which embedding model to use
        idx = Path(index_path)
        meta_path = idx / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            model_name = meta.get("model_name", fallback_model)
            logger.info("Loaded embedding model: %s", model_name)
        else:
            model_name = fallback_model
            logger.warning("No metadata.json found, falling back to: %s", model_name)

        # 2) Instantiate embeddings and load FAISS store
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.store = FAISS.load_local(
            index_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("FAISS index loaded from %s", index_path)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Runs a similarity search, applies a softmax over the returned scores,
        and returns each hit with a [prob=XX.XX%] confidence and a valid path.
        """
        docs_and_scores = self.store.similarity_search_with_relevance_scores(
            query, k=top_k
        )

        raw_scores = np.array([score for _, score in docs_and_scores], dtype=np.float32)
        exp_scores = np.exp(raw_scores)
        probs = exp_scores / exp_scores.sum()

        results: List[Dict] = []
        for (doc, _), prob in zip(docs_and_scores, probs):
            # Try metadata first
            path = doc.metadata.get('path')
            # Fallback to chunks.json mapping if missing
            if not path or path is None:
                content_key = doc.page_content.strip()
                path = self.chunk_map.get(content_key)
            # As last resort, show unknown
            if not path:
                path = ['<unknown>']

            print(f"Confidence: {prob:.2%}  Path: {' > '.join(path)}")
            results.append({
                'path': path,
                'content': doc.page_content,
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

Απαντάς στην ερώτηση/προσταγή: «{question}» ελέγχοντας και ακολουθώντας τους κανόνες. Αν υπάρχει σύγκρουση, κάνε ασφαλή ολοκλήρωση.
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

            # build contexts safely
            contexts: List[str] = []
            for d in raw:
                p = d.get('path')
                if isinstance(p, (list, tuple)):
                    path_str = ' > '.join(p)
                elif isinstance(p, str):
                    path_str = p
                else:
                    path_str = ''
                contexts.append(f"{path_str}\n{d['content']}")
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
    tree = TreeRAGPipeline(cfg.faiss_index_path, cfg.hf_embedding_model)
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
