#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
from sentence_transformers import SentenceTransformer
from langchain.schema import Document


from preprocessing.PDFParser import PDFParser
from preprocessing.Chunker import Chunker
from preprocessing.AdvancedChunker import AdvancedChunker
from preprocessing.TreeChunker import TreeChunker  # <— make sure this path is correct

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PDF → Markdown → Chunks → Embeddings → FAISS"
    )
    parser.add_argument(
        "input_pdf",
        type=Path,
        help="Path to the input PDF file"
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("preprocessing/document_edited.md"),
        help="Intermediate Markdown output"
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("preprocessing/chunks.json"),
        help="Intermediate JSON chunks file"
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("faiss_index"),
        help="Output FAISS index directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lighteternal/stsb-xlm-r-greek-transfer",
        help="SentenceTransformer model name"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Threads per worker (PDF → MD converter)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Worker processes (PDF → MD & chunker)"
    )
    parser.add_argument(
        "--chunker-type",
        choices=["chunker", "advanced", "tree"],
        default="advanced",
        help="Which chunker to use"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Merge threshold for TreeChunker only"
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def main():
    args = parse_args()
    setup_logging()

    # 1) Convert PDF → Markdown (uncomment if needed)
    # logger.info("Converting PDF %s → Markdown %s", args.input_pdf, args.markdown)
    # pdf_converter = PDFParser(
    #     num_threads=args.threads,
    #     max_workers=args.workers
    # )
    # pdf_converter.convert(args.input_pdf, args.markdown)

    # 2) Chunk Markdown → JSON
    logger.info(
        "Chunking Markdown %s → JSON %s using %s chunker",
        args.markdown, args.chunks, args.chunker_type
    )

    if args.chunker_type == "chunker":
        chunker = Chunker(
            md_path=args.markdown,
            out_json=args.chunks,
            processes=args.workers
        )
    elif args.chunker_type == "advanced":
        chunker = AdvancedChunker(
            md_path=args.markdown,
            out_json=args.chunks,
            processes=args.workers
        )
    else:  # tree
        chunker = TreeChunker(
            md_path=args.markdown,
            out_json=args.chunks,
            merge_threshold=args.threshold,
            embed_model_name=args.model,
            batch_size=args.workers  # or keep your original default
        )

    chunker.run()

    # -------------------------------------------------------------------------
    # 3a) If using TreeChunker, build & save a raw FAISS index directly
    # -------------------------------------------------------------------------
    if args.chunker_type == "tree":
        logger.info("Building FAISS index with TreeChunker.build_index()")

        # build the tree‐chunker’s own index (it uses SentenceTransformer internally)
        chunker.embedder = SentenceTransformer(args.model)
        chunker.build_index()

        # 1) Turn your chunks into LangChain Documents with metadata:

        docs = [
            Document(
                page_content=meta["content"],
                metadata={"path": meta["path"]}
            )
            for meta in chunker.metadata
        ]

        # 2) Build a LangChain FAISS store using a LangChain embedding wrapper

        hf_embedder = HuggingFaceEmbeddings(model_name=args.model)
        vs = FAISS.from_documents(docs, hf_embedder)

        # save model_name & dim
        meta = {
            "model_name": args.model,
            "dim": chunker.index.d
        }
        with (args.index / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # 3) And *then* save the LangChain store locally:
        args.index.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(args.index))

        logger.info("✅ FAISS index (with metadata) saved to %s", args.index)
        return

    # -------------------------------------------------------------------------
    # 3b) Otherwise, fall back to LangChain's FAISS.from_texts
    # -------------------------------------------------------------------------
    logger.info("Loading %s", args.chunks)
    with args.chunks.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [chunk["data"] for chunk in chunks]

    logger.info("Embedding %d chunks with model %s", len(texts), args.model)
    embedder = HuggingFaceEmbeddings(model_name=args.model)
    vs = FAISS.from_texts(texts, embedder)

    args.index.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(args.index))
    logger.info("✅ LangChain FAISS index saved to %s", args.index)

    # 4) Persist metadata so query-time uses the same model & dim
    test_vec = embedder.embed_query("test")
    meta = {
        "model_name": args.model,
        "dim": len(test_vec)
    }
    with (args.index / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info(
        "Wrote metadata.json with model_name=%s and dim=%d",
        meta["model_name"], meta["dim"]
    )


if __name__ == "__main__":
    main()