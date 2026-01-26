import pandas as pd
import pdfplumber
import json
from pathlib import Path

from utils.pdf_extraction import extract_text_two_columns
from utils.text_cleaning import (
    clean_text, remove_headers_footers, remove_references
)
from utils.chunking import create_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
OUTPUT_PATH = PROJECT_ROOT / "data" / "chunks.json"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"

metadata = pd.read_csv(METADATA_PATH)
chunks = []

for i in range(len(metadata)):
    pdf_path = PAPERS_DIR / f"{metadata.paper_id.iloc[i]}.pdf"
    pages_lines = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = extract_text_two_columns(page)
            pages_lines.append(page_text.split("\n"))

    pages_lines = remove_headers_footers(pages_lines)

    text = "\n\n".join("\n".join(lines) for lines in pages_lines)
    text = remove_references(text)
    text = clean_text(text)

    chunks.extend(create_chunks(text, metadata, i))

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)
