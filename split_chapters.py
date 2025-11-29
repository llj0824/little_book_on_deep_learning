#!/usr/bin/env python3
"""
Split "The Little Book on Deep Learning" into per-chapter PDFs and text files.

Outputs a folder per chapter containing `chapter.pdf` (page subset) and
`chapter.txt` (plain text). Page numbers in CHAPTERS are 1-based and inclusive.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pdfplumber
from pypdf import PdfReader, PdfWriter


@dataclass(frozen=True)
class Chapter:
    order: int
    title: str
    slug: str
    start_page: int  # 1-based inclusive
    end_page: int  # 1-based inclusive


# Major sections of the book and their page ranges.
CHAPTERS: List[Chapter] = [
    Chapter(order=0, title="Foreword", slug="foreword", start_page=8, end_page=9),
    Chapter(order=1, title="Machine Learning", slug="machine_learning", start_page=11, end_page=19),
    Chapter(order=2, title="Efficient Computation", slug="efficient_computation", start_page=20, end_page=24),
    Chapter(order=3, title="Training", slug="training", start_page=25, end_page=57),
    Chapter(order=4, title="Model Components", slug="model_components", start_page=58, end_page=97),
    Chapter(order=5, title="Architectures", slug="architectures", start_page=98, end_page=116),
    Chapter(order=6, title="Prediction", slug="prediction", start_page=117, end_page=137),
    Chapter(order=7, title="Synthesis", slug="synthesis", start_page=138, end_page=145),
    Chapter(order=8, title="The Compute Schism", slug="the_compute_schism", start_page=146, end_page=157),
    Chapter(order=9, title="The missing bits", slug="the_missing_bits", start_page=158, end_page=163),
    Chapter(order=10, title="Bibliography", slug="bibliography", start_page=164, end_page=175),
    Chapter(order=11, title="Index", slug="index", start_page=176, end_page=185),
]


def ensure_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return cleaned or "chapter"


def load_selected_chapters(only: Iterable[str] | None) -> List[Chapter]:
    if not only:
        return CHAPTERS
    wanted = {ensure_slug(item) for item in only}
    return [chapter for chapter in CHAPTERS if chapter.slug in wanted]


def extract_text(book_path: Path, chapter: Chapter) -> str:
    with pdfplumber.open(book_path) as pdf:
        parts: List[str] = []
        for page_index in range(chapter.start_page - 1, chapter.end_page):
            page = pdf.pages[page_index].dedupe_chars()
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            parts.append(f"--- Page {page_index + 1} ---\n{text.strip()}\n")
    return "\n".join(parts).strip() + "\n"


def write_pdf_subset(reader: PdfReader, chapter: Chapter, output_path: Path) -> None:
    writer = PdfWriter()
    for page_index in range(chapter.start_page - 1, chapter.end_page):
        writer.add_page(reader.pages[page_index])
    with output_path.open("wb") as f:
        writer.write(f)


def split_book(book_path: Path, output_dir: Path, selected: List[Chapter]) -> None:
    reader = PdfReader(str(book_path))
    total_pages = len(reader.pages)

    for chapter in selected:
        if chapter.start_page < 1 or chapter.end_page > total_pages:
            raise ValueError(
                f"Chapter {chapter.title} pages [{chapter.start_page}, {chapter.end_page}] "
                f"fall outside the PDF (1-{total_pages})."
            )
        if chapter.start_page > chapter.end_page:
            raise ValueError(f"Chapter {chapter.title} has start > end.")

        folder = output_dir / f"{chapter.order:02d}_{chapter.slug}"
        folder.mkdir(parents=True, exist_ok=True)

        pdf_path = folder / "chapter.pdf"
        text_path = folder / "chapter.txt"

        print(f"Writing {chapter.title} -> {pdf_path}")
        write_pdf_subset(reader, chapter, pdf_path)

        print(f"Extracting text -> {text_path}")
        text = extract_text(book_path, chapter)
        text_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split the book into per-chapter PDFs and text.")
    parser.add_argument(
        "--book-path",
        default="little_book_on_deep_learning.pdf",
        type=Path,
        help="Path to the source PDF.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("chapters"),
        type=Path,
        help="Directory where chapter folders will be written.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Optional list of chapter slugs to process (e.g., foundations deep_models).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = load_selected_chapters(args.only)
    if not selected:
        raise SystemExit("No chapters matched --only selection.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_book(args.book_path, args.output_dir, selected)


if __name__ == "__main__":
    main()
