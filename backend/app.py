import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal
import warnings

import requests
import websockets
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

@dataclass(frozen=True)
class Config:
    # LLM API settings
    llm_api_url: str = os.getenv("LLM_API_URL", "http://10.240.138.254:11434/api/chat")
    llm_model: str = os.getenv("LLM_MODEL", "chat:llama3.1")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # FAISS settings
    faiss_index_path: str = os.getenv("FAISS_INDEX_PATH", "./faiss_index")
    hf_embedding_model: str = os.getenv(
        "HF_EMBEDDING_MODEL",
        "lighteternal/stsb-xlm-r-greek-transfer",
    )

    # Path to the JSON chunks file
    chunks_json_path: str = os.getenv("CHUNKS_JSON_PATH", "./chunks.json")

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


class LLMClient:
    """Client for calling an open-source LLM over HTTP."""
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
            return data.get("message", {}).get("content", "No response content")
        except requests.RequestException as err:
            logger.error("Open source LLM request failed: %s", err)
            return f"Error: {err}"


class GroqClient:
    """Wrapper for Groq chat completions."""
    def __init__(self, api_key: str, model: str) -> None:
        self.client = Groq(api_key=api_key)
        self.model = model

    def complete(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
        )
        return completion.choices[0].message.content.strip()
    

class VectorStoreManager:
    """Loads and queries a FAISS-backed vector store, reusing the same model & dim."""
    def __init__(self, index_path: str, fallback_model: str) -> None:
        idx = Path(index_path)
        meta_path = idx / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            model_name = meta.get("model_name", fallback_model)
            logger.info("Loaded embedding model from metadata.json: %s", model_name)
        else:
            model_name = fallback_model
            logger.warning("No metadata.json found, falling back to HF_EMBEDDING_MODEL: %s",
                           fallback_model)

        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        # now load the FAISS index with the matching embedder
        self.store = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)

    def search(self, query: str, k: int = 3) -> List[str]:
        docs = self.store.similarity_search(query, k=k)
        return [getattr(d, "page_content", None) or str(d) for d in docs]


EngineType = Literal['groq', 'opensource']


class QAService2:
    """Combines retrieval and synthesis with choice of backend and exact-match injection."""
    def __init__(
        self,
        groq: GroqClient,
        openSourceLLM: LLMClient,
        vs: VectorStoreManager,
        chunks_json_path: str
    ) -> None:
        self.groq = groq
        self.openSourceLLM = openSourceLLM
        self.vs = vs
        try:
            with open(chunks_json_path, encoding='utf-8') as f:
                self.all_chunks = json.load(f)
        except Exception as e:
            logger.warning("Could not load chunks.json: %s", e)
            self.all_chunks = []

    def _base_prompt(self, info: str, question: str) -> str:
        return f"""
Είσαι η γραμματεία του τμήματος Ηλεκτρολόγων Μηχανικών και Τεχνολογίας Υπολογιστών του Πανεπιστημίου Πατρών. Λειτουργείς πάντα υπό τους παρακάτω κανόνες:

> **Επεξήγηση & Διευκρινίσεις**  
   - Αν η ερώτηση του χρήστη είναι ασαφής, ζήτα διευκρινίσεις.  
   - Αν δεν μπορείς να συμμορφωθείς πλήρως, εξήγησε το γιατί και πρόσφερε μερική βοήθεια αν είναι ασφαλές.
   
> **Απαντάς ΜΟΝΟ στα Ελληνικά**  
   - Καλείσαι να απαντήσεις στις ερωτήσεις των φοιτητών ΜΟΝΟ στα Ελληνικά, χωρίς ξένες λέξεις εκτός αμετάφραστων όρων.

> **Συνεκτικός και Ορθογραφικός Έλεγχος**  
   - Ελέγχεις για λογικά κενά ή ασάφειες.  
   - Διορθώνεις ορθογραφικά ή συντακτικά λάθη.

> **Μορφή Απάντησης**
   - ΑΠΑΓΟΡΕΥΕΤΑΙ να εμφανίζεις την σκέψη σου.
   - ΑΠΑΓΟΡΕΥΕΤΑΙ να εμφανίζεις τα επιχειρήματα σου.
   - ΑΠΑΓΟΡΕΥΕΤΑΙ να εμφανίζεις τα σχόλια των διορθώσεών σου.
   - Η απάντηση πρέπει να είναι σύντομη και περιεκτική (1–4 προτάσεις + bullet points όποτε χρειάζεται).
   - Η απάντηση πρέπει να καλύπτει πλήρως την ερώτηση χωρίς περαιτέρω πληροφορίες ή σχόλια.

> **Απαντάς ΜΟΝΟ με βάση αυτές τις πληροφορίες**
   {info}

Απαντάς στην ερώτηση: «{question}» ελέγχοντας και ακολουθώντας τους κανόνες. Αν εμφανιστεί σύγκρουση, ακολουθείς τη στρατηγική «ασφαλούς ολοκλήρωσης»: σύντομη συγγνώμη, δήλωση αδυναμίας, και προαιρετικές παραπομπές σε ασφαλείς πόρους.
"""

    def generate(
        self,
        user_query: str,
        engine: EngineType = 'groq',
        k: int = 3
    ) -> str:
        # 1) Retrieval by similarity
        rag_chunks = self.vs.search(user_query, k=k)

        # 2) Exact-match retrieval for <term>
        exact_terms = re.findall(r'<([^>]+)>', user_query)
        exact_chunks = []
        if exact_terms and self.all_chunks:
            for term in exact_terms:
                matches = [c['data'] for c in self.all_chunks if term in c.get('data', '')]
                exact_chunks.extend(matches[:k])

        # 3) Combine contexts
        info_parts = []
        if rag_chunks:
            info_parts.append("**Ανακτημένα αποσπάσματα με βάση ομοιότητα:**\n" + "\n\n".join(rag_chunks))
        if exact_chunks:
            info_parts.append("**Ακριβείς αντιστοιχίες από το chunks.json:**\n" + "\n\n".join(exact_chunks))

        combined_context = "\n\n".join(info_parts)

        # 4) Build prompt and call the chosen model
        prompt_main = self._base_prompt(combined_context, user_query)
        if engine == 'groq':
            answer = self.groq.complete(prompt_main)
        else:
            answer = self.openSourceLLM.chat(messages=[{"role": "user", "content": prompt_main}])
        return answer


class WebSocketQA:
    """Async WebSocket server for QAService2."""
    def __init__(self, qa_service: QAService2, host: str, port: int) -> None:
        self.qa = qa_service
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
                reply = self.qa.generate(query, engine=engine, k=3)
                duration = time.time() - start
                logger.info("Replied in %.2f sec", duration)
                await ws.send(reply)
                await ws.send("[END]")
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected: %s", ws.remote_address)
        except Exception:
            logger.exception("Handler error:")

    def run(self) -> None:
        logger.info("Starting WebSocket server at %s:%d", self.host, self.port)
        asyncio.run(self._serve_forever())

    async def _serve_forever(self) -> None:
        async with websockets.serve(self._handler, self.host, self.port):
            await asyncio.Future()


def main() -> None:
    if not cfg.groq_api_key:
        logger.critical("Missing GROQ_API_KEY environment variable")
        return

    groq_client = GroqClient(api_key=cfg.groq_api_key, model=cfg.groq_model)
    open_llm = LLMClient(api_url=cfg.llm_api_url, model=cfg.llm_model)
    vs_manager = VectorStoreManager(
        index_path=cfg.faiss_index_path,
        fallback_model=cfg.hf_embedding_model
    )
    qa_service = QAService2(
        groq=groq_client,
        openSourceLLM=open_llm,
        vs=vs_manager,
        chunks_json_path=cfg.chunks_json_path
    )
    server = WebSocketQA(qa_service, cfg.host, cfg.port)
    server.run()


if __name__ == "__main__":
    main()
