# =============================================================================
# rag/splitter.py
# =============================================================================
# WHAT THIS FILE DOES:
#   Takes the raw text extracted from a PDF and breaks it into smaller,
#   overlapping "chunks" that can be embedded and stored in the vector DB.
#
# WHY DO WE CHUNK?
#   LLMs have a context window limit (max tokens they can read at once).
#   Also, embedding a 50-page doc as one giant blob would be useless —
#   you'd always retrieve the entire doc no matter what you asked.
#   Chunking lets us retrieve only the RELEVANT pieces for each question.
#
# WHY OVERLAP?
#   If a sentence spans the boundary between two chunks, without overlap
#   it would get cut in half and lose meaning. Overlap ensures no idea
#   is lost at the seams between chunks.
#
#   Example with chunk_size=20, overlap=5:
#   Text:    "The quick brown fox jumps over the lazy dog"
#   Chunk 1: "The quick brown fox"
#   Chunk 2: "fox jumps over the"   ← "fox" repeated from chunk 1
#   Chunk 3: "the lazy dog"
# =============================================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Splits raw text into overlapping chunks using LangChain's
    RecursiveCharacterTextSplitter.

    HOW RecursiveCharacterTextSplitter WORKS:
    ------------------------------------------
    It tries a list of separators IN ORDER, and falls back to the next
    one if a split would produce a chunk that's still too large.

    Default separator priority:
        1. "\\n\\n"  → paragraph breaks (best semantic boundary)
        2. "\\n"     → line breaks
        3. " "       → word boundaries
        4. ""        → individual characters (last resort)

    This is "recursive" because it keeps trying smaller separators
    until every chunk fits within chunk_size. This preserves semantic
    meaning much better than a naive "split every N chars" approach.

    Args:
        text (str):           The full raw text extracted from the PDF.
        chunk_size (int):     Max number of characters per chunk.
                              1000 chars ≈ ~250 tokens (1 token ≈ 4 chars).
        chunk_overlap (int):  How many characters to repeat between
                              consecutive chunks to preserve context.

    Returns:
        List[Document]: A list of LangChain Document objects.
                        Each Document has:
                          .page_content → the chunk text
                          .metadata     → dict (empty here, can store page #)
    """

    # --- Build the splitter ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,       # max chars per chunk
        chunk_overlap=chunk_overlap, # chars of overlap between chunks
        length_function=len,         # how to measure chunk size (char count)
        separators=[                 # try these in order
            "\n\n",   # paragraphs  ← most preferred
            "\n",     # lines
            ". ",     # sentences   ← added for better semantic splitting
            " ",      # words
            "",       # characters  ← absolute last resort
        ],
        is_separator_regex=False,    # treat separators as plain strings, not regex
    )

    # --- Create Document objects ---
    # create_documents() takes a list of strings (we pass one big string).
    # It returns a list of Document objects, one per chunk.
    #
    # A LangChain Document is just a simple wrapper:
    #   Document(page_content="chunk text here", metadata={...})
    #
    # We use Documents (not raw strings) because the FAISS vector store
    # and LangChain retrievers all expect this format — it lets metadata
    # travel with each chunk (e.g., source file name, page number, etc.)
    documents = splitter.create_documents([text])

    return documents


def split_text_with_metadata(
    text: str,
    source_name: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Same as split_text() but attaches metadata (source filename) to
    every chunk. Useful for multi-PDF setups where you want to know
    WHICH document a chunk came from.

    The metadata dict travels with the chunk all the way into FAISS,
    so when you retrieve a chunk you also get its source info.

    Args:
        text (str):        Full raw text from the PDF.
        source_name (str): The filename or identifier for this document.
        chunk_size (int):  Max chars per chunk.
        chunk_overlap (int): Overlap chars between chunks.

    Returns:
        List[Document]: Chunks with metadata = {"source": source_name}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )

    # Pass metadata per document — each chunk inherits this metadata dict
    documents = splitter.create_documents(
        texts=[text],
        metadatas=[{"source": source_name}],  # one metadata dict per text string
    )

    return documents


def get_chunk_stats(documents: List[Document]) -> dict:
    """
    Helper that returns stats about the chunks for display in the UI.
    Useful for debugging chunk size settings.

    Args:
        documents (List[Document]): The chunked documents.

    Returns:
        dict: Stats including count, avg size, min/max size.
    """
    if not documents:
        return {"count": 0, "avg_chars": 0, "min_chars": 0, "max_chars": 0}

    sizes = [len(doc.page_content) for doc in documents]

    return {
        "count":     len(documents),
        "avg_chars": int(sum(sizes) / len(sizes)),
        "min_chars": min(sizes),
        "max_chars": max(sizes),
    }
