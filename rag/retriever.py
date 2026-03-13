# =============================================================================
# rag/retriever.py
# =============================================================================
# WHAT THIS FILE DOES:
#   Given a user's question, finds the most semantically similar chunks
#   from the FAISS vector store and returns them as context for the LLM.
#
# HOW SIMILARITY SEARCH WORKS:
#   1. The user's question is embedded into a vector (same model used for chunks)
#   2. FAISS computes the cosine similarity between the query vector and
#      every stored chunk vector
#   3. The top-K closest chunks (by similarity score) are returned
#
# COSINE SIMILARITY:
#   Two vectors point in "similar directions" if they represent similar meaning.
#   Cosine similarity = 1.0 means identical, 0.0 means completely unrelated.
#   FAISS does this comparison in milliseconds even for millions of vectors.
#
# WHY NOT JUST KEYWORD SEARCH (like grep)?
#   Keyword search misses synonyms and paraphrasing.
#   Semantic search understands that "What is the penalty for late payment?"
#   and "What happens if I pay after the due date?" mean the same thing.
# =============================================================================

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List, Tuple


def get_relevant_chunks(
    query: str,
    vector_store: FAISS,
    k: int = 4,
) -> List[Document]:
    """
    Finds the top-k most semantically similar chunks for a given query.

    Under the hood, LangChain does this:
        1. Calls the same embedding model used during ingestion on your query
        2. Gets a query vector, e.g. [0.21, -0.83, 0.44, ...]
        3. Asks FAISS to find the k nearest vectors in the index
        4. Returns the original Document objects that correspond to those vectors

    Args:
        query (str):              The user's question or search string.
        vector_store (FAISS):     A loaded FAISS vector store instance.
        k (int):                  Number of top chunks to retrieve.
                                  4 is a good default — enough context
                                  without flooding the LLM prompt.

    Returns:
        List[Document]: Top-k most relevant Document chunks.
                        Each has .page_content (the text) and .metadata.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",   # plain cosine similarity
        search_kwargs={"k": k},
    )

    # .invoke() is the standard LCEL interface — replaces the old .get_relevant_documents()
    results = retriever.invoke(query)
    return results


def get_relevant_chunks_with_scores(
    query: str,
    vector_store: FAISS,
    k: int = 4,
) -> List[Tuple[Document, float]]:
    """
    Same as get_relevant_chunks() but also returns the similarity score
    for each chunk. Scores are useful for:
      - Debugging: seeing HOW similar a chunk is to the query
      - Filtering:  discarding chunks below a score threshold
      - UI display: showing confidence indicators

    FAISS returns L2 distance by default when using similarity_search_with_score.
    Lower L2 distance = more similar (distance 0 = identical).
    If you used cosine similarity during index creation, scores are cosine distances.

    Args:
        query (str):          The user's question.
        vector_store (FAISS): A loaded FAISS vector store.
        k (int):              Number of top results to return.

    Returns:
        List[Tuple[Document, float]]: List of (document, score) pairs,
                                      sorted by score ascending (most similar first).
    """
    results_with_scores = vector_store.similarity_search_with_score(query, k=k)
    return results_with_scores


def get_relevant_chunks_filtered(
    query: str,
    vector_store: FAISS,
    k: int = 4,
    score_threshold: float = 1.0,
) -> List[Document]:
    """
    Retrieves top-k chunks but filters out any with a similarity score
    ABOVE the threshold (since FAISS L2 distance — lower = better).

    Use this when you want to avoid returning irrelevant chunks if the
    query topic doesn't exist in the document at all.

    For example, if someone asks "What is the capital of France?" and your
    PDF is an employee handbook, all scores will be high (bad matches) —
    this filter would return nothing, letting the chain say "I don't know."

    Args:
        query (str):             The user's question.
        vector_store (FAISS):    A loaded FAISS vector store.
        k (int):                 Max number of top results to consider.
        score_threshold (float): Max L2 distance to allow. Chunks with
                                 distance > threshold are discarded.
                                 Tune this based on your embedding model.
                                 A typical good value is 0.8–1.2 for L2.

    Returns:
        List[Document]: Filtered list of relevant Document chunks.
    """
    results_with_scores = vector_store.similarity_search_with_score(query, k=k)

    filtered = [
        doc
        for doc, score in results_with_scores
        if score <= score_threshold
    ]

    return filtered


def format_retrieved_chunks(documents: List[Document]) -> str:
    """
    Formats a list of retrieved Document chunks into a single string
    that can be inserted into an LLM prompt as context.

    Each chunk is separated by a divider so the LLM can distinguish
    between different parts of the document.

    Args:
        documents (List[Document]): The retrieved chunks.

    Returns:
        str: A formatted string ready to be used as prompt context.
             Returns a fallback message if no documents were retrieved.
    """
    if not documents:
        return "No relevant context found in the document."

    parts = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Document")
        header = f"[Chunk {i} | Source: {source}]"
        parts.append(f"{header}\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(parts)
