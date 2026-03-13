import fitz  # PyMuPDF — imported as "fitz" (its legacy name)
import os
from typing import Optional


def load_pdf(file_path: str) -> str:
    """
    Opens a PDF file and extracts all text from every page.

    How it works:
    - fitz.open() loads the PDF into memory
    - We iterate over every page object
    - page.get_text() pulls the raw text from that page
    - We prepend a [Page N] label so the LLM (and you) know where text came from
    - All pages are joined into one big string and returned

    Args:
        file_path: Absolute or relative path to the .pdf file

    Returns:
        A single string containing all extracted text, with page markers.

    Raises:
        FileNotFoundError: If the PDF path doesn't exist.
        ValueError: If the file is not a valid PDF or is empty.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found at path: {file_path}")

    if not file_path.lower().endswith(".pdf"):
        raise ValueError(f"File must be a .pdf — got: {file_path}")

    doc = fitz.open(file_path)

    if doc.page_count == 0:
        raise ValueError("The PDF has no pages.")

    full_text_parts = []

    for page_num in range(doc.page_count):
        page = doc[page_num]

        # get_text("text") extracts plain text.
        # Other modes: "html", "dict", "blocks" — we use plain text for simplicity.
        page_text = page.get_text("text")

        # Strip leading/trailing whitespace per page
        page_text = page_text.strip()

        if page_text:  # skip blank pages
            full_text_parts.append(f"[Page {page_num + 1}]\n{page_text}")

    doc.close()

    if not full_text_parts:
        raise ValueError("No text could be extracted from the PDF. It may be scanned/image-based.")

    # Join all pages with double newlines so the splitter can find paragraph boundaries
    full_text = "\n\n".join(full_text_parts)

    return full_text


def get_pdf_metadata(file_path: str) -> dict:
    """
    Extracts metadata from a PDF (title, author, page count, etc.)
    Useful for displaying info in the UI sidebar.

    Args:
        file_path: Path to the .pdf file

    Returns:
        A dict with keys like 'title', 'author', 'page_count', 'file_name'
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found at path: {file_path}")

    doc = fitz.open(file_path)
    meta = doc.metadata  # built-in PyMuPDF metadata dict

    result = {
        "file_name": os.path.basename(file_path),
        "file_size_kb": round(os.path.getsize(file_path) / 1024, 2),
        "page_count": doc.page_count,
        "title": meta.get("title") or "Unknown",
        "author": meta.get("author") or "Unknown",
        "subject": meta.get("subject") or "Unknown",
        "creator": meta.get("creator") or "Unknown",
    }

    doc.close()
    return result
