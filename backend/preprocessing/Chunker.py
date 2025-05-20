import re
import json
import logging
import multiprocessing
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple, Pattern, Match

# ─── Logging Setup ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Chunker")


class Chunker:
    """
    Splits a Markdown document into smaller chunks around H1–H3 headers,
    rewrites any Markdown tables to include only their nearest context,
    and outputs the result as a JSON array of {"id", "data"} objects.
    """

    # Regex patterns
    _TABLE_RE: Pattern = re.compile(r'(?sm)(^.*?\n)?(\|.*?\|(?:\n\|.*?\|)+)(\n.*?$)?')
    _HEADER_RE: Pattern = re.compile(r'(?m)^#{1,3} .+')

    # Class‐shared text buffer for worker processes
    _text: str

    def __init__(
        self,
        md_path: Path,
        out_json: Path,
        processes: int | None = None,
        table_context: bool = True,
    ) -> None:
        self.md_path = md_path
        self.out_json = out_json
        self.processes = processes or multiprocessing.cpu_count()
        self.table_context = table_context

        logger.debug(
            "Initialized with md_path=%s, out_json=%s, processes=%d, table_context=%s",
            md_path, out_json, self.processes, table_context
        )

    @classmethod
    def _init_worker(cls, md_path_str: str) -> None:
        """
        Worker initializer: load the entire Markdown into a class variable.
        Avoids pickling large buffers on each task.
        """
        cls._text = Path(md_path_str).read_text(encoding='utf-8')
        logger.debug("Worker %d loaded text (%d chars)", multiprocessing.current_process().pid, len(cls._text))

    @staticmethod
    def _process_table_chunk(match: Match) -> str:
        """
        Rewrite a Markdown table plus its nearest non-empty lines.
        """
        text = match.group(0)
        lines = text.splitlines()
        # find table line indices
        idxs = [i for i, l in enumerate(lines) if l.startswith('|') and l.endswith('|')]
        if not idxs:
            return text

        start, end = idxs[0], idxs[-1] + 1
        before = next((ln.strip() for ln in reversed(lines[:start]) if ln.strip()), "")
        after  = next((ln.strip() for ln in lines[end:] if ln.strip()), "")
        chunk_lines = lines[start:end]
        # filter out any empty lines from context
        return "\n".join(filter(None, [before, *chunk_lines, after]))

    @classmethod
    def _custom_chunk(cls, bounds: Tuple[int, int]) -> List[str]:
        """
        Worker function:
         1) Slice the loaded text by codepoint‐safe indices.
         2) Optionally rewrite tables.
         3) Split on H1–H3 headers.
        """
        start, end = bounds
        section = cls._text[start:end]

        # rewrite tables if enabled
        if cls._TABLE_RE:
            section = cls._TABLE_RE.sub(cls._process_table_chunk, section)

        # find header positions
        points = [m.start() for m in cls._HEADER_RE.finditer(section)] + [len(section)]
        chunks: List[str] = []
        for i in range(len(points) - 1):
            chunk = section[points[i]:points[i + 1]].strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def run(self) -> None:
        """
        Main entrypoint: scans headers, divides into slices,
        fans out to worker pool, flattens, and writes JSON.
        """
        logger.info("Reading and scanning headers in %s", self.md_path)
        full_text = self.md_path.read_text(encoding='utf-8')
        header_positions = [m.start() for m in self._HEADER_RE.finditer(full_text)]
        boundaries = [0] + header_positions + [len(full_text)]
        slices = [
            (boundaries[i], boundaries[i + 1])
            for i in range(len(boundaries) - 1)
        ]

        n_slices = len(slices)
        chunksize = max(1, n_slices // (self.processes * 4))
        logger.info(
            "Chunking %d sections across %d processes (chunksize=%d)",
            n_slices, self.processes, chunksize
        )

        with Pool(
            processes=self.processes,
            initializer=type(self)._init_worker,
            initargs=(str(self.md_path),)
        ) as pool:
            batches = pool.map(type(self)._custom_chunk, slices, chunksize)

        # flatten the list of lists
        flat_chunks = [chunk for batch in batches for chunk in batch]
        logger.info("Produced %d chunks; writing to %s", len(flat_chunks), self.out_json)

        with self.out_json.open("w", encoding="utf-8") as jf:
            json.dump(
                [{"id": idx, "data": data} for idx, data in enumerate(flat_chunks)],
                jf,
                ensure_ascii=False,
                indent=2
            )
        logger.info("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Split a Markdown document into header-based chunks (and simplify tables)."
    )
    parser.add_argument(
        "input_md",
        type=Path,
        help="Path to the source Markdown file"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("chunks.json"),
        help="Destination JSON file for chunks"
    )
    parser.add_argument(
        "-p", "--processes",
        type=int,
        default=None,
        help="Number of worker processes (defaults to CPU count)"
    )
    args = parser.parse_args()

    if not args.input_md.is_file():
        logger.error("Input file %s does not exist.", args.input_md)
        exit(1)

    chunker = Chunker(
        md_path=args.input_md,
        out_json=args.output,
        processes=args.processes
    )
    chunker.run()
