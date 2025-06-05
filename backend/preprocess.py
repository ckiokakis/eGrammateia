#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from preprocessing.PDFParser import PDFParser
from preprocessing.TreeChunker import TreeChunker
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process PDF → Markdown → TreeChunked Chunks → FAISS index"
    )
    parser.add_argument("--input_pdf", type=Path, help="Path to the input PDF file")
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("preprocessing/document_edited.md"),
        help="Intermediate Markdown output file"
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
        "--threshold",
        type=float,
        default=0.8,
        help="Merge threshold for TreeChunker"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Threads per worker for the conversion pipeline",
    )
    parser.add_argument(
        "--download",
        type=bool,
        default=False,
        help="Download the study guide again"
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

    if args.download:
        downloader = StudyGuideDownloader(
            base_url="https://www.ece.upatras.gr/index.php/el/curriculum.html",
            download_dir="docs",
            verify_ssl=False  # Only needed if you have certificate issues
        )

        downloaded_file_path = downloader.download_first_study_guide()
        logger.info(f"Saved to: {downloaded_file_path}")
    
    if input_pdf or args.download:
        pdf_path =  downloaded_file_path if args.download and downloaded_file_path else args.input_pdf

        logger.info("Converting PDF %s → Markdown %s", pdf_path, args.markdown)
        pdf_converter = PDFParser(
            num_threads=args.threads,
            max_workers=args.workers
        )
        pdf_converter.convert(pdf_path, args.markdown)

        logger.info(
            "Chunking Markdown %s → JSON %s using TreeChunker",
            args.markdown, args.chunks
        )

    chunker = TreeChunker(
        md_path=args.markdown,
        out_json=args.chunks,
        merge_threshold=args.threshold,
        embed_model_name=args.model,
        batch_size=args.workers
    )

    chunker.run()

    logger.info("Building FAISS index with TreeChunker")

    chunker.embedder = SentenceTransformer(args.model)
    chunker.build_index()

    docs = [
        Document(
            page_content=meta["content"],
            metadata={"path": meta["path"]}
        )
        for meta in chunker.metadata
    ]

    hf_embedder = HuggingFaceEmbeddings(model_name=args.model)
    vs = FAISS.from_documents(docs, hf_embedder)

    args.index.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(args.index))

    meta = {
        "model_name": args.model,
        "dim": chunker.index.d
    }
    with (args.index / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("✅ FAISS index and metadata saved to %s", args.index)


if __name__ == "__main__":
    main()
