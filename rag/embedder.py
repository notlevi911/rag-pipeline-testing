# =============================================================================
# rag/embedder.py
# =============================================================================
# WHAT THIS FILE DOES:
#   1. Defines a custom LangChain Embeddings class (GeminiEmbeddings) that
#      uses the google-genai SDK (google.genai.Client) directly — no wrapper.
#   2. Takes chunked Document objects and converts them into float vectors.
#   3. Stores those vectors in a FAISS index (in memory + on disk).
#   4. Provides helpers to save/load/delete the FAISS index from disk so we
#      don't have to re-embed the entire PDF on every app restart.
#
# WHY USE google.genai.Client DIRECTLY?
#   langchain-google-genai v2 routes embedding calls through the v1beta API
#   endpoint, which doesn't support text-embedding-004. The google-genai SDK
#   calls the v1 endpoint directly, so text-embedding-004 works fine.
#
# HOW THE CUSTOM EMBEDDINGS CLASS PLUGS INTO LANGCHAIN:
#   LangChain's FAISS.from_documents() only needs an object that implements
#   two methods:
#       embed_documents(texts: List[str]) -> List[List[float]]
#       embed_query(text: str)            -> List[float]
#   Our GeminiEmbeddings class inherits from langchain_core.embeddings.Embeddings
#   and implements exactly those two methods using google.genai.Client.
#   FAISS doesn't care how the vectors are produced — it just needs the numbers.
#
# WHAT IS AN EMBEDDING?
#   An embedding is a list of floating-point numbers that encodes the semantic
#   meaning of a piece of text in high-dimensional space.
#
#   Example — text-embedding-004 produces 768-dimensional vectors:
#     "The cat sat on the mat"   → [0.021, -0.834,  0.412, ... 768 numbers]
#     "A feline rested on a rug" → [0.019, -0.829,  0.408, ... 768 numbers]
#     "The stock market crashed" → [0.721,  0.334, -0.910, ... 768 numbers]
#
#   The first two are close together in vector space (similar meaning).
#   The third is far away. This is what makes semantic search work.
#
# WHAT IS FAISS?
#   FAISS (Facebook AI Similarity Search) is a C++ library with Python bindings
#   that stores vectors and finds nearest neighbours in milliseconds, even with
#   millions of vectors. LangChain wraps it so you work with Documents, not raw
#   numpy arrays.
#
# FILES WRITTEN TO DISK:
#   vector_store/index.faiss  → the raw float vectors in binary FAISS format
#   vector_store/index.pkl    → pickled dict mapping vector IDs → Documents
# =============================================================================

import os
from typing import List, Optional

from google import genai
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VECTOR_STORE_PATH = "vector_store"
FAISS_INDEX_FILE  = os.path.join(VECTOR_STORE_PATH, "index.faiss")
FAISS_PKL_FILE    = os.path.join(VECTOR_STORE_PATH, "index.pkl")


# =============================================================================
# CUSTOM EMBEDDINGS CLASS
# =============================================================================

class GeminiEmbeddings(Embeddings):
    """
    A LangChain-compatible Embeddings implementation that calls the Google
    Generative AI embedding API using google.genai.Client directly.

    Why subclass langchain_core.embeddings.Embeddings?
    ---------------------------------------------------
    LangChain's FAISS, retrievers, and other vector store components accept
    any object that has embed_documents() and embed_query(). By inheriting
    from the base Embeddings class we signal the correct interface and get
    a few free utilities (like async fallback wrappers) at no cost.

    Usage:
        emb = GeminiEmbeddings()                          # reads GOOGLE_API_KEY from env
        emb = GeminiEmbeddings(model="text-embedding-004")
        vector = emb.embed_query("What is the refund policy?")
        vectors = emb.embed_documents(["chunk one text", "chunk two text"])
    """

    def __init__(
        self,
        model:   str            = "text-embedding-004",
        api_key: Optional[str]  = None,
    ) -> None:
        """
        Args:
            model (str):            The Google embedding model to use.
                                    "text-embedding-004" gives 768-dim vectors
                                    and is available free via AI Studio.
            api_key (str, optional): Your Google API key. If omitted, reads
                                     from the GOOGLE_API_KEY environment variable.
        """
        self.model = model

        resolved_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Google API key not found. Set GOOGLE_API_KEY in your .env file "
                "or pass api_key= to GeminiEmbeddings()."
            )

        # google.genai.Client is the entry point for the google-genai SDK.
        # We use the default v1beta endpoint — that's where the available
        # embedding models (gemini-embedding-001, gemini-embedding-2-preview)
        # actually live on this account.
        self.client = genai.Client(api_key=resolved_key)

    # ------------------------------------------------------------------
    # Required by LangChain's Embeddings interface
    # ------------------------------------------------------------------

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document chunks into float vectors.

        Called by FAISS.from_documents() during ingestion.
        Each chunk in `texts` gets its own vector in the returned list.

        HOW IT WORKS:
            client.models.embed_content() sends all texts to the Gemini
            Embeddings API in one request. The response contains one
            ContentEmbedding per input text, each with a .values list
            of floats (768 numbers for text-embedding-004).

        Args:
            texts (List[str]): The chunk texts to embed. Can be 1 to many.

        Returns:
            List[List[float]]: One float vector per input text.
                               Shape: (len(texts), 768)
        """
        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,       # pass the full list — SDK batches it
        )
        # response.embeddings → List[ContentEmbedding]
        # each ContentEmbedding has a .values attribute → List[float]
        return [embedding.values for embedding in response.embeddings]

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string into a float vector.

        Called at query time to turn the user's question into a vector
        so FAISS can compare it against all stored chunk vectors.

        IMPORTANT: You MUST use the same model here as you used during
        ingestion (embed_documents). Mixing models produces incompatible
        vector spaces and similarity search will return garbage results.

        Args:
            text (str): The user's question or search string.

        Returns:
            List[float]: A single float vector of length 768.
        """
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,        # single string
        )
        return response.embeddings[0].values


# =============================================================================
# FACTORY — get a ready-to-use GeminiEmbeddings instance
# =============================================================================

def get_embeddings(model: Optional[str] = None) -> GeminiEmbeddings:
    """
    Returns a configured GeminiEmbeddings instance.

    Reads the model name from the EMBEDDING_MODEL env var if set,
    otherwise falls back to "text-embedding-004".

    Args:
        model (str, optional): Override the embedding model name.

    Returns:
        GeminiEmbeddings: Ready to use for embed_documents / embed_query.
    """
    embedding_model = model or os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    return GeminiEmbeddings(model=embedding_model)


# =============================================================================
# VECTOR STORE CREATION
# =============================================================================

def create_vector_store(
    documents: List[Document],
    model:     Optional[str] = None,
) -> FAISS:
    """
    Embeds a list of Document chunks and stores them in a FAISS index.
    Also saves the index to disk for reuse across app restarts.

    WHAT HAPPENS INSIDE FAISS.from_documents():
    --------------------------------------------
    1. Calls embeddings.embed_documents([doc.page_content for doc in documents])
       → our GeminiEmbeddings.embed_documents() fires, hitting the Google API
    2. Receives back a list of float vectors (one per chunk)
    3. Builds an in-memory FAISS index from those vectors
    4. Stores the mapping: vector ID → original Document object
       (so similarity_search returns the text, not just a vector ID)

    Args:
        documents (List[Document]): Chunked documents from splitter.py.
        model (str, optional):      Embedding model override.

    Returns:
        FAISS: In-memory vector store, also persisted to disk.

    Raises:
        ValueError: If documents list is empty.
    """
    if not documents:
        raise ValueError("Cannot create vector store: documents list is empty.")

    embeddings = get_embeddings(model)

    # Core call — embeds every chunk and builds the FAISS index in one shot
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    # Persist to disk:
    #   index.faiss → raw binary vectors
    #   index.pkl   → pickled {vector_id: Document} mapping
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    vector_store.save_local(VECTOR_STORE_PATH)

    return vector_store


# =============================================================================
# VECTOR STORE LOADING
# =============================================================================

def load_vector_store(model: Optional[str] = None) -> FAISS:
    """
    Loads a previously saved FAISS index from disk back into memory.

    WHY allow_dangerous_deserialization=True?
    -----------------------------------------
    FAISS uses Python's pickle (.pkl) to store document texts alongside
    the vectors. Pickle can execute arbitrary code on load, so LangChain v0.2+
    makes you explicitly opt in with this flag. Since we wrote the .pkl file
    ourselves during create_vector_store(), it is safe to load.

    CRITICAL: The embedding model passed here MUST match the one used when
    the index was built. If you change EMBEDDING_MODEL, delete the
    vector_store/ folder and re-process your PDF.

    Args:
        model (str, optional): Embedding model name. Must match what was
                               used during create_vector_store().

    Returns:
        FAISS: The restored vector store, ready for similarity search.

    Raises:
        FileNotFoundError: If no saved index exists on disk.
    """
    if not vector_store_exists():
        raise FileNotFoundError(
            f"No saved vector store found at '{VECTOR_STORE_PATH}/'. "
            "Please process a PDF first to build the index."
        )

    embeddings = get_embeddings(model)

    return FAISS.load_local(
        folder_path=VECTOR_STORE_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,   # we trust our own pickle file
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def vector_store_exists() -> bool:
    """
    Returns True if both FAISS index files exist on disk.
    Used by the UI to decide whether to show a "Load Previous Index" button.
    """
    return (
        os.path.exists(FAISS_INDEX_FILE)
        and os.path.exists(FAISS_PKL_FILE)
    )


def delete_vector_store() -> None:
    """
    Deletes the saved FAISS index from disk and recreates an empty folder.
    Called when the user clicks "New PDF" in the sidebar.
    """
    import shutil
    if os.path.exists(VECTOR_STORE_PATH):
        shutil.rmtree(VECTOR_STORE_PATH)
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)


def add_documents_to_store(
    new_documents: List[Document],
    model:         Optional[str] = None,
) -> FAISS:
    """
    Merges new document chunks into an existing FAISS index.
    Useful for a multi-PDF setup where you keep adding documents.

    HOW IT WORKS:
        1. Load the existing index from disk
        2. Embed the new documents into a temporary FAISS store
        3. Call existing_store.merge_from(new_store) to combine them
        4. Save the merged store back to disk

    Args:
        new_documents (List[Document]): New chunks to add.
        model (str, optional):          Must match the existing index's model.

    Returns:
        FAISS: The updated combined vector store.
    """
    if not vector_store_exists():
        return create_vector_store(new_documents, model)

    embeddings = get_embeddings(model)

    existing_store = FAISS.load_local(
        folder_path=VECTOR_STORE_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    new_store = FAISS.from_documents(
        documents=new_documents,
        embedding=embeddings,
    )

    # merge_from() adds all vectors from new_store into existing_store in-place
    existing_store.merge_from(new_store)
    existing_store.save_local(VECTOR_STORE_PATH)

    return existing_store


def get_store_info(vector_store: FAISS) -> dict:
    """
    Returns basic info about the vector store for sidebar display.

    Args:
        vector_store (FAISS): A loaded or freshly created vector store.

    Returns:
        dict: total_chunks, index_path, index_exists.
    """
    doc_count = len(vector_store.docstore._dict)

    return {
        "total_chunks": doc_count,
        "index_path":   os.path.abspath(VECTOR_STORE_PATH),
        "index_exists": vector_store_exists(),
    }
