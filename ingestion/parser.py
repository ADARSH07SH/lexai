"""
Document parsing and chunking pipeline.

Handles:
  - Native PDF text extraction (PyPDF2)
  - OCR fallback for scanned PDFs (pytesseract + pypdfium2)
  - Heading/body classification
  - Recursive text splitting into retrieval-ready chunks
"""

import os
from PyPDF2 import PdfReader
import pytesseract
import pypdfium2 as pdfium
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Auto-detect Tesseract on Windows
if os.name == "nt":
    _tesseract_candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(
            os.getenv("USERNAME", "")
        ),
    ]
    for _path in _tesseract_candidates:
        if os.path.exists(_path):
            pytesseract.pytesseract.tesseract_cmd = _path
            break


def read_pdf(uploaded_file) -> str:
    """Extract text from a native (text-based) PDF."""
    content = ""
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        content += page.extract_text() or ""
    return content


def ocr_pdf(uploaded_file) -> str:
    """Extract text from a scanned PDF via OCR (Tesseract)."""
    extracted = ""
    pdf = pdfium.PdfDocument(uploaded_file)
    for page_idx in range(len(pdf)):
        page = pdf.get_page(page_idx)
        bitmap = page.render(scale=1, rotation=0)
        image = bitmap.to_pil()
        try:
            extracted += pytesseract.image_to_string(
                image, config=r"--oem 3 --psm 6"
            )
        except pytesseract.TesseractNotFoundError:
            print("[LexAI] Tesseract not found — falling back to native PDF reader.")
            return read_pdf(uploaded_file)
    return extracted


def _detect_heading(line: str) -> bool:
    """Return True if a line looks like a section heading (all-caps)."""
    return line.isupper()


def chunk_document(raw_text: str) -> list:
    """
    Split raw document text into tagged, retrieval-ready chunks.

    1. Groups consecutive lines under their nearest heading.
    2. Splits each group body with RecursiveCharacterTextSplitter.
    3. Returns a list of dicts: { heading, body, documents }.
    """
    sections = []
    current_heading = None
    current_body = ""

    for line in raw_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        if _detect_heading(line):
            # Save the previous section before starting a new one
            if current_heading is not None:
                sections.append(
                    {"heading": current_heading, "body": current_body.strip()}
                )
            current_heading = line
            current_body = ""
        else:
            current_body += " " + line

    # Flush the last section
    if current_heading is not None:
        sections.append(
            {"heading": current_heading, "body": current_body.strip()}
        )

    # Split each section body into smaller documents
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n", "\n\n", "\n", " "],
        chunk_size=1000,
        chunk_overlap=30,
    )
    for section in sections:
        section["documents"] = splitter.create_documents([section["body"]])

    return sections
