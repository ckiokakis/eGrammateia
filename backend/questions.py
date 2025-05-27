import json
import time
from pathlib import Path

from app2 import Config, LLMClient, GroqClient, VectorStoreManager, TreeRAGPipeline, QAService2

def main():
    # 1) load configuration & clients
    cfg = Config()
    groq = GroqClient(api_key=cfg.groq_api_key, model=cfg.groq_model)
    llm  = LLMClient(api_url=cfg.llm_api_url, model=cfg.llm_model)
    vs   = VectorStoreManager(cfg.faiss_index_path, cfg.hf_embedding_model)
    tree = TreeRAGPipeline(
        index_path=cfg.faiss_index_path,
        llm=llm,
        fallback_model=cfg.hf_embedding_model,
        chunks_json_path=cfg.chunks_json_path
    )

    # choose tree‐RAG; if you want flat, switch use_tree to False
    qa = QAService2(
        groq=groq,
        llm=llm,
        vs=vs,
        tree=tree,
        use_tree=True,
        chunks_json_path=cfg.chunks_json_path
    )

    # 2) load all questions
    qpath = Path("docs/test_questions.json")
    questions = json.loads(qpath.read_text(encoding="utf-8"))

    all_answers = []
    for question in questions:
        # 3) retrieve with confidences
        retrieved = tree.retrieve(question, top_k=5)

        # 4) generate (with retry on timeout)
        while True:
            answer = qa.generate(question, engine="groq", k=5)
            if not answer.startswith("Error: HTTPConnectionPool"):
                break
            print(f"Timeout for question {question!r}, retrying in 5s…")
            time.sleep(5)

        # 5) collect into our output format
        all_answers.append({
            "question": question,
            "retrieved_context": retrieved,
            "answer": answer.strip()
        })

        # be polite
        time.sleep(1)

    # 6) write to answers.json
    out_path = Path("answers_k5_Llama31.json")
    out_path.write_text(json.dumps(all_answers, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_answers)} answers to {out_path!r}")

if __name__ == "__main__":
    main()
