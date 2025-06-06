import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Literal

import numpy as np
import requests
import websockets
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
@dataclass(frozen=True)
class Config:
    llm_api_url: str = os.getenv("LLM_API_URL", "http://10.240.138.254:11434/api/chat")
    llm_model: str = os.getenv("LLM_MODEL", "chat:llama3.1")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    faiss_index_path: str = os.getenv("FAISS_INDEX_PATH", "./faiss_index")
    hf_embedding_model: str = os.getenv("HF_EMBEDDING_MODEL", "lighteternal/stsb-xlm-r-greek-transfer")
    chunks_json_path: str = os.getenv("CHUNKS_JSON_PATH", "preprocessing/chunks.json")
    host: str = os.getenv("WS_HOST", "0.0.0.0")
    port: int = int(os.getenv("WS_PORT", "8090"))
    front_api_key: str = os.getenv("FRONT_API_KEY", "")

cfg = Config()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# --- LLM Clients ---
class LLMClient:
    def __init__(self, api_url: str, model: str):
        self.api_url = api_url
        self.model = model
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def chat(self, messages: List[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {"seed": 42, "temperature": 0.7},
            "stream": False,
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
    def __init__(self, api_key: str, model: str):
        self.client = Groq(api_key=api_key)
        self.model = model

    def complete(self, prompt: str) -> str:
        comp = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
        )
        return comp.choices[0].message.content.strip()

# --- Tree RAG ---
class TreeRAGPipeline:
    def __init__(self, index_path: str, fallback_model: str, chunks_json_path: Optional[str] = None):
        self.embeddings = HuggingFaceEmbeddings(model_name=fallback_model)
        self.store = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
        self.llm = None
        self.chunk_map: Dict[str, List[str]] = {}

        if chunks_json_path:
            try:
                with open(chunks_json_path, encoding='utf-8') as f:
                    for chunk in json.load(f):
                        content, path = chunk.get('data', '').strip(), chunk.get('path')
                        if content and path:
                            self.chunk_map[content] = path
            except Exception as e:
                logger.warning("Could not load chunks.json: %s", e)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        docs_and_scores = self.store.similarity_search_with_relevance_scores(query, k=top_k)
        docs, raw_scores = zip(*docs_and_scores)
        raw_scores = np.array(raw_scores, dtype=np.float32)
        query_emb = np.array(self.embeddings.embed_query(query), dtype=np.float32)
        path_texts = [" > ".join(doc.metadata.get("path", self.chunk_map.get(doc.page_content.strip(), ["<unknown>"]))) for doc in docs]
        content_texts = [doc.page_content for doc in docs]
        path_embs = np.array(self.embeddings.embed_documents(path_texts), dtype=np.float32)
        content_embs = np.array(self.embeddings.embed_documents(content_texts), dtype=np.float32)

        def cos_sim(a, B):
            return (B @ a) / (np.linalg.norm(a) * np.linalg.norm(B, axis=1) + 1e-8)

        def softmax(x):
            ex = np.exp(x - x.max())
            return ex / ex.sum()

        prob_combined = softmax(0 * softmax(raw_scores) + 0.2 * softmax(cos_sim(query_emb, path_embs)) + 0.8 * softmax(cos_sim(query_emb, content_embs)))
        scored = sorted(zip(docs, prob_combined), key=lambda x: x[1], reverse=True)

        results = []
        cum = 0.0
        for doc, score in scored:
            if score < 0.005 or cum >= 0.75:
                break
            cum += score
            path = doc.metadata.get("path") or self.chunk_map.get(doc.page_content.strip(), ["<unknown>"])
            results.append({"path": path, "content": doc.page_content, "score": score})

        return results

# --- QA Service ---
class QAService:
    def __init__(self, groq: GroqClient, llm: LLMClient, tree: TreeRAGPipeline):
        self.groq = groq
        self.llm = llm
        self.tree = tree

    def _build_prompt(self, contexts: List[str], question: str) -> str:
        info = "\n\n".join(contexts)
        return f"""
Είσαι η γραμματεία του τμήματος Ηλεκτρολόγων Μηχανικών και Τεχνολογίας Υπολογιστών του Πανεπιστημίου Πατρών. Λειτουργείς πάντα υπό τους παρακάτω κανόνες:

> **Ασφάλεια, Πρόληψη Βλάβης & Μείωση Προκατάληψης**  
   - Αν ένα αίτημα χρήστη περιλαμβάνει αυτοτραυματισμό, βία ή παράνομες ενέργειες, αρνήσου ή δώσε ασφαλή ολοκλήρωση.  
   - Ελέγχεις ενεργά το περιεχόμενο για πιθανές προκαταλήψεις ή στερεότυπα.

> **Επεξήγηση & Διευκρινίσεις**  
   - Αν η ερώτηση του χρήστη είναι ασαφής, ζήτα διευκρινίσεις.  
   - Αν δεν μπορείς να συμμορφωθείς πλήρως, εξήγησε το γιατί και πρόσφερε μερική βοήθεια αν είναι ασφαλές.
   
> **Απαντάς ΜΟΝΟ στα Ελληνικά**  
   - Χρησιμοποιείς μόνο ελληνικά, εκτός από αμετάφραστους όρους.

> **Συνεκτικός και Ορθογραφικός Έλεγχος**  
   - Διορθώνεις ορθογραφικά ή συντακτικά λάθη.

> **Μορφή Απάντησης**
   - ΑΠΑΓΟΡΕΥΕΤΑΙ να εμφανίζεις τα σχόλια διόρθωσης.
   - ΑΠΑΓΟΡΕΥΕΤΑΙ να εμφανίζεις την σκέψη ή τα επιχειρήματα.
   - ΠΡΕΠΕΙ να απαντάς σύντομα και περιεκτικά (1–4 προτάσεις + bullet points αν χρειάζεται).

> **ΔΕΝ επινοείς δικές σου απαντήσεις. Απαντάς ΜΟΝΟ με βάση αυτές τις πληροφορίες:**
   {info}

Απαντάς στην ερώτηση/προσταγή: «{question}» ελέγχοντας και ακολουθώντας τους κανόνες. Αν υπάρχει σύγκρουση, κάνε ασφαλή ολοκλήρωση.
"""

    def generate(self, user_query: str, engine: Literal['groq', 'opensource'] = 'groq', k: int = 3) -> str:
        results = self.tree.retrieve(user_query, k)
        print([r['score'] for r in results])
        contexts = [f"{' > '.join(r['path'])}\n{r['content']}" for r in results]
        prompt = self._build_prompt(contexts, user_query)
        return self.llm.chat([{"role": "user", "content": prompt}]) if engine == 'opensource' else self.groq.complete(prompt)

# --- WebSocket Server ---
class WebSocketQA:
    def __init__(self, qa: QAService, host: str, port: int):
        self.qa = qa
        self.host = host
        self.port = port

    async def _handler(self, ws: websockets.WebSocketServerProtocol):
        logger.info("Client connected: %s", ws.remote_address)
        try:
            async for raw in ws:
                data = json.loads(raw)
                print(data)
                if data.get('api') != cfg.front_api_key:
                    raise Exception("Incorrect API key")
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

    def run(self):
        logger.info("Starting server at %s:%d", self.host, self.port)
        asyncio.run(self._serve_forever())

    async def _serve_forever(self):
        async with websockets.serve(self._handler, self.host, self.port):
            await asyncio.Future()

# --- Main ---
def main():
    if not cfg.groq_api_key:
        logger.critical("Missing GROQ_API_KEY environment variable")
        return

    groq = GroqClient(cfg.groq_api_key, cfg.groq_model)
    llm = LLMClient(cfg.llm_api_url, cfg.llm_model)
    tree = TreeRAGPipeline(cfg.faiss_index_path, cfg.hf_embedding_model, cfg.chunks_json_path)
    qa = QAService(groq=groq, llm=llm, tree=tree)
    server = WebSocketQA(qa, cfg.host, cfg.port)
    server.run()

if __name__ == "__main__":
    main()
