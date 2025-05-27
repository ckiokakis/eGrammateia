#!/usr/bin/env python3
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

class TreeChunker:
    """
    Splits a Markdown file into a hierarchy of (path, text) chunks,
    optionally merges similar adjacent chunks, and writes out JSON.
    """

    def __init__(
        self,
        md_path: Path,
        out_json: Path,
        merge_threshold: float = 0.8,
        embed_model_name: str = 'lighteternal/stsb-xlm-r-greek-transfer',
        batch_size: int = 32
    ):
        self.md_path = md_path
        self.out_json = out_json
        self.merge_threshold = merge_threshold
        self.embed_model_name = embed_model_name
        self.batch_size = batch_size

    def run(self) -> None:
        # 1) Parse the Markdown into flat list of {'path': [...], 'data': str}
        nodes = self._parse_markdown()
        texts = [n['data'] for n in nodes]

        # 2) Load the model once and batch‐encode all texts
        embedder = SentenceTransformer(self.embed_model_name)
        # convert_to_tensor=False so we get numpy arrays back
        embs = embedder.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_tensor=False,
            show_progress_bar=True
        )
        embs = np.asarray(embs, dtype='float32')
        # normalize for cosine similarity
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs /= np.where(norms == 0, 1e-8, norms)

        # 3) Merge adjacent chunks if similarity >= threshold
        merged: List[Dict] = []
        for idx, node in enumerate(nodes):
            if not merged:
                merged.append(dict(node))
                continue

            # compute cosine similarity to last merged chunk
            sim = float(np.dot(embs[idx], embs[len(merged) - 1]))
            if sim >= self.merge_threshold:
                merged[-1]['data'] += "\n\n" + node['data']
            else:
                merged.append(dict(node))

        # 4) Write out JSON
        with open(self.out_json, 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        print(f"✅ Wrote {self.out_json} (threshold={self.merge_threshold}, batch_size={self.batch_size})")

    def _parse_markdown(self) -> List[Dict]:
        with open(self.md_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        nodes: List[Dict] = []
        stack: List[tuple] = [(-1, [])]  # (header_level, path_list)
        buf: List[str] = []

        def flush():
            if buf:
                _, path = stack[-1]
                nodes.append({
                    'path': path.copy(),
                    'data': ''.join(buf).strip()
                })
                buf.clear()

        for line in lines:
            m = re.match(r'^(#{1,6})\s+(.*)', line)
            if m:
                flush()
                level = len(m.group(1))
                title = m.group(2).strip()
                # pop until we're at the parent level
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, stack[-1][1] + [title]))
            else:
                buf.append(line)
        flush()
        return nodes
    
    def build_index(self) -> None:
        """Combine path and content, embed, normalize, and build FAISS index."""
        with open("preprocessing/chunks.json", 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        texts = []
        meta = []
        for node in chunks:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Markdown into tree chunks.")
    parser.add_argument("input_md", type=Path, help="Path to the source Markdown file")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("chunks_tree.json"),
        help="Destination JSON file for chunks"
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.8,
        help="Cosine similarity threshold for merging (0–1)"
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=32,
        help="Batch size for embedding (tune to your RAM)"
    )
    args = parser.parse_args()

    if not args.input_md.is_file():
        print(f"ERROR: Input file {args.input_md} does not exist.")
        exit(1)

    chunker = TreeChunker(
        md_path=args.input_md,
        out_json=args.output,
        merge_threshold=args.threshold,
        batch_size=args.batch_size
    )
    chunker.run()
