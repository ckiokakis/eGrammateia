import argparse
import logging
import os
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import torch
from PyPDF2 import PdfReader, PdfWriter
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from docling.datamodel.base_models import InputFormat

# module‐level logger
logger = logging.getLogger(__name__)

# this will live in each worker process
_worker_converter: DocumentConverter | None = None


class PDFParser:
    """
    Convert a multi-page PDF into a single Markdown document,
    processing pages in parallel across multiple worker processes.
    """

    def __init__(
        self,
        device: AcceleratorDevice | None = None,
        num_threads: int | None = None,
        max_workers: int | None = None,
    ):
        # choose device
        self.device = device or (
            AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU
        )
        # threads per worker
        self.num_threads = num_threads or os.cpu_count() or 1
        # number of parallel workers
        self.max_workers = max_workers or os.cpu_count() or 1

        # build per-process converter options
        pdf_opts = PdfPipelineOptions()
        pdf_opts.accelerator_options = AcceleratorOptions(
            device=self.device,
            num_threads=self.num_threads,
        )
        self.format_options = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)
        }

        logger.debug(
            "Converter initialized with device=%s, threads=%d, workers=%d",
            self.device,
            self.num_threads,
            self.max_workers,
        )

    @staticmethod
    def init_worker():
        """
        ProcessPoolExecutor initializer: builds a DocumentConverter
        in each worker process.
        """
        global _worker_converter
        device = AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU
        pdf_opts = PdfPipelineOptions()
        pdf_opts.accelerator_options = AcceleratorOptions(
            device=device,
            num_threads=os.cpu_count() or 1,
        )
        _worker_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
        )
        logger.debug("Worker %d initialized converter on %s", os.getpid(), device)

    @staticmethod
    def convert_page_worker(page_pdf_path: str) -> str:
        """
        Worker function: convert a single‐page PDF to Markdown.
        """
        global _worker_converter
        if _worker_converter is None:
            PDFParser.init_worker()

        result = _worker_converter.convert(Path(page_pdf_path))
        return result.document.export_to_markdown()

    def split_to_pages(self, pdf_path: Path, output_dir: Path) -> list[Path]:
        """
        Split each page of `pdf_path` into its own single‐page PDF
        under `output_dir`, returning their paths.
        """
        reader = PdfReader(str(pdf_path))
        output_dir.mkdir(parents=True, exist_ok=True)
        page_files: list[Path] = []

        logger.info("Splitting '%s' into %d pages…", pdf_path, len(reader.pages))
        for idx, page in enumerate(reader.pages, start=1):
            page_path = output_dir / f"page_{idx}.pdf"
            writer = PdfWriter()
            writer.add_page(page)
            with open(page_path, "wb") as f:
                writer.write(f)
            page_files.append(page_path)
            logger.debug("Wrote single‐page PDF %s", page_path)

        return page_files

    def convert(self, pdf_path: Path, out_md: Path) -> None:
        """
        Convert the PDF at `pdf_path` into Markdown, streaming
        each page in parallel and writing to `out_md`.
        """
        logger.info("Starting conversion of '%s' → '%s'", pdf_path, out_md)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            page_files = self.split_to_pages(pdf_path, tmpdir_path)

            logger.info(
                "Converting %d pages with %d worker(s)…",
                len(page_files),
                self.max_workers,
            )
            with open(out_md, "w", encoding="utf-8") as md_file, ProcessPoolExecutor(
                max_workers=self.max_workers,
                initializer=PDFParser.init_worker,
            ) as executor:
                for page_md in executor.map(
                    PDFParser.convert_page_worker,
                    map(str, page_files),
                ):
                    md_file.write(page_md)
                    md_file.write("\n\n")
                    logger.debug("Wrote Markdown for one page")

        logger.info("Conversion complete. Markdown saved to '%s'", out_md)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a PDF document to Markdown in parallel."
    )
    parser.add_argument(
        "input_pdf", type=Path, help="Path to the input PDF file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("document.md"),
        help="Path for the output Markdown file",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Threads per worker for the conversion pipeline",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes",
    )
    args = parser.parse_args()

    # configure root logger
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger.setLevel(logging.DEBUG)

    if not args.input_pdf.is_file():
        logger.error("Input file '%s' does not exist.", args.input_pdf)
        exit(1)

    converter = PDFParser(
        num_threads=args.threads, max_workers=args.workers
    )
    converter.convert(args.input_pdf, args.output)


if __name__ == "__main__":
    main()
