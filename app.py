# =============================================================================
# app.py — Main Streamlit Application
# =============================================================================
# WHAT THIS FILE DOES:
#   This is the entry point for the entire RAG chatbot app.
#   Run it with: streamlit run app.py
#
# HOW STREAMLIT WORKS (important to understand first!):
#   Streamlit is unusual compared to normal Python scripts.
#   Every time the user interacts with ANYTHING (clicks a button, types in
#   a text box, uploads a file), Streamlit RE-RUNS THE ENTIRE SCRIPT from
#   top to bottom.
#
#   This means all your local variables get reset on every interaction!
#   To persist data between re-runs, you use st.session_state — a special
#   dict that survives re-runs for the duration of the browser session.
#
#   Things we persist in session_state:
#     - messages:       the full chat history (list of dicts)
#     - chain:          the built LangChain LCEL chain (expensive to rebuild)
#     - pdf_processed:  whether a PDF has been ingested (bool)
#     - pdf_name:       the name of the currently loaded PDF (str)
#     - chunk_stats:    stats about the chunks for display (dict)
#
# MODELS USED:
#   - Chat:       gemini-2.5-flash  (FREE via Google AI Studio)
#   - Embeddings: models/text-embedding-004  (FREE via Google AI Studio)
#   Get your key at: https://aistudio.google.com/app/apikey
# =============================================================================

import os
import streamlit as st
from dotenv import load_dotenv

# Load .env file FIRST — before importing anything that needs GOOGLE_API_KEY
load_dotenv()

from rag.loader import load_pdf, get_pdf_metadata
from rag.splitter import split_text_with_metadata, get_chunk_stats
from rag.embedder import (
    create_vector_store,
    load_vector_store,
    vector_store_exists,
    delete_vector_store,
    get_store_info,
)
from rag.chain import build_chain, get_answer_stream, build_chat_history, trim_chat_history


# =============================================================================
# PAGE CONFIGURATION
# Must be the very first Streamlit call in the script.
# =============================================================================
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CUSTOM CSS
# Streamlit allows injecting raw CSS via st.markdown with unsafe_allow_html.
# We use this to style the chat messages and header area.
# =============================================================================
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p  { margin: 0.3rem 0 0 0; opacity: 0.85; font-size: 0.95rem; }

    /* Chunk/stats cards */
    .stat-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
    }

    /* Source chunk expander styling */
    .chunk-box {
        background: #f0f4ff;
        border-left: 3px solid #667eea;
        padding: 0.8rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.82rem;
        color: #333;
        margin: 0.3rem 0;
    }

    /* Pipeline status steps */
    .step-done  { color: #28a745; font-weight: 600; }
    .step-wait  { color: #adb5bd; }
    .step-error { color: #dc3545; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALISATION
# =============================================================================
# These keys are set once on first run and persist across all re-runs.
# Always guard with `if "key" not in st.session_state` so they're only
# initialised once, not reset on every re-run.

if "messages"       not in st.session_state:
    st.session_state.messages = []          # chat history: [{"role": ..., "content": ...}]

if "chain"          not in st.session_state:
    st.session_state.chain = None           # the built LangChain chain

if "pdf_processed"  not in st.session_state:
    st.session_state.pdf_processed = False  # has a PDF been ingested?

if "pdf_name"       not in st.session_state:
    st.session_state.pdf_name = None        # name of the currently loaded PDF

if "chunk_stats"    not in st.session_state:
    st.session_state.chunk_stats = None     # {"count": N, "avg_chars": N, ...}

if "pdf_metadata"   not in st.session_state:
    st.session_state.pdf_metadata = None    # {"title": ..., "author": ..., ...}


# =============================================================================
# HELPER: Check API Key
# =============================================================================

def check_api_key() -> bool:
    """Returns True if GOOGLE_API_KEY is set, False otherwise."""
    key = os.getenv("GOOGLE_API_KEY", "")
    return bool(key and key.strip() and key != "your-google-api-key-here")


# =============================================================================
# HELPER: Ingest PDF
# Runs the full ingestion pipeline and updates session state.
# We break this into a function to keep the main script clean.
# =============================================================================

def ingest_pdf(uploaded_file) -> bool:
    """
    Runs the full ingestion pipeline for an uploaded PDF:
        1. Save PDF bytes to disk
        2. Extract text with PyMuPDF
        3. Split into chunks with RecursiveCharacterTextSplitter
        4. Embed chunks and store in FAISS
        5. Build the LangChain chain
        6. Update session state

    Returns:
        bool: True on success, False on error.
    """
    try:
        # --- Step 0: Save uploaded bytes to disk ---
        # st.file_uploader gives us a BytesIO-like object.
        # PyMuPDF needs an actual file path, so we save it first.
        os.makedirs("uploads", exist_ok=True)
        pdf_path = os.path.join("uploads", uploaded_file.name)

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # --- Step 1: Extract text ---
        step1 = st.empty()
        step1.info("📖 **Step 1/4** — Extracting text from PDF...")
        raw_text = load_pdf(pdf_path)
        meta = get_pdf_metadata(pdf_path)
        step1.success(f"✅ **Step 1/4** — Extracted text from **{meta['page_count']} pages**")

        # --- Step 2: Split into chunks ---
        step2 = st.empty()
        chunk_size    = int(os.getenv("CHUNK_SIZE", "1000"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))

        step2.info(f"✂️ **Step 2/4** — Splitting text (chunk_size={chunk_size}, overlap={chunk_overlap})...")
        documents = split_text_with_metadata(
            text=raw_text,
            source_name=uploaded_file.name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        stats = get_chunk_stats(documents)
        step2.success(
            f"✅ **Step 2/4** — Created **{stats['count']} chunks** "
            f"(avg {stats['avg_chars']} chars each)"
        )

        # --- Step 3: Embed and store in FAISS ---
        step3 = st.empty()
        step3.info("🧠 **Step 3/4** — Embedding chunks and building FAISS index...")
        step3.info(
            "🧠 **Step 3/4** — Calling Google Embeddings API... "
            f"(embedding {stats['count']} chunks, this may take a moment)"
        )
        vector_store = create_vector_store(documents)
        store_info = get_store_info(vector_store)
        step3.success(
            f"✅ **Step 3/4** — FAISS index built with **{store_info['total_chunks']} vectors** "
            f"and saved to disk"
        )

        # --- Step 4: Build the LangChain chain ---
        step4 = st.empty()
        step4.info("⛓️ **Step 4/4** — Building LangChain LCEL chain...")
        chain = build_chain(vector_store)
        step4.success("✅ **Step 4/4** — Chain ready! You can now ask questions.")

        # --- Update session state ---
        st.session_state.chain         = chain
        st.session_state.pdf_processed = True
        st.session_state.pdf_name      = uploaded_file.name
        st.session_state.chunk_stats   = stats
        st.session_state.pdf_metadata  = meta
        st.session_state.messages      = []  # reset chat for the new PDF

        return True

    except FileNotFoundError as e:
        st.error(f"❌ File error: {e}")
        return False
    except ValueError as e:
        st.error(f"❌ PDF processing error: {e}")
        return False
    except Exception as e:
        st.error(f"❌ Unexpected error during ingestion: {e}")
        return False


# =============================================================================
# SIDEBAR
# The sidebar contains: API key check, PDF uploader, ingestion controls,
# PDF metadata, chunk stats, and a "load previous index" button.
# =============================================================================

with st.sidebar:
    st.header("⚙️ Setup")

    # --- API Key Status ---
    if check_api_key():
        st.success("🔑 Google API key loaded", icon="✅")
    else:
        st.error(
            "🔑 **Google API key missing!**\n\n"
            "Create a `.env` file in the project root with:\n"
            "`GOOGLE_API_KEY=your-key-here`\n\n"
            "Get a free key at: https://aistudio.google.com/app/apikey",
            icon="❌"
        )
        st.stop()  # halt the app — nothing works without the key

    st.divider()

    # --- PDF Upload Section ---
    st.subheader("📤 Upload PDF")
    uploaded_file = st.file_uploader(
        label="Choose a PDF file",
        type=["pdf"],
        help="Upload any PDF document. Text-based PDFs work best. "
             "Scanned/image PDFs may not extract text correctly.",
    )

    if uploaded_file is not None:
        # Show file info before processing
        file_size_kb = round(len(uploaded_file.getbuffer()) / 1024, 1)
        st.caption(f"📎 `{uploaded_file.name}` — {file_size_kb} KB")

        if st.button("🚀 Process PDF", type="primary", use_container_width=True):
            with st.spinner("Running RAG ingestion pipeline..."):
                # We call this from outside the sidebar context so the
                # progress messages appear in the main area, not the sidebar.
                pass  # actual call happens in main area below

            # Store the uploaded file reference for processing in main area
            st.session_state["_pending_upload"] = uploaded_file

            # Use st.rerun() to re-render with the pending upload flag set
            st.rerun()

    # --- Load Previous Index ---
    if not st.session_state.pdf_processed and vector_store_exists():
        st.divider()
        st.subheader("📂 Previous Index")
        st.caption("A saved FAISS index was found on disk.")

        if st.button("Load Previous Index", use_container_width=True):
            with st.spinner("Loading saved FAISS index..."):
                try:
                    vector_store = load_vector_store()
                    st.session_state.chain         = build_chain(vector_store)
                    st.session_state.pdf_processed = True
                    st.session_state.pdf_name      = "Previously indexed document"
                    store_info = get_store_info(vector_store)
                    st.session_state.chunk_stats   = {
                        "count":     store_info["total_chunks"],
                        "avg_chars": "N/A",
                        "min_chars": "N/A",
                        "max_chars": "N/A",
                    }
                    st.success("✅ Previous index loaded!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load index: {e}")

    # --- Current PDF Info ---
    if st.session_state.pdf_processed:
        st.divider()
        st.subheader("📄 Current Document")
        st.markdown(f"**{st.session_state.pdf_name}**")

        if st.session_state.pdf_metadata:
            meta = st.session_state.pdf_metadata
            with st.expander("Document Metadata"):
                st.markdown(f"- **Pages:** {meta.get('page_count', 'N/A')}")
                st.markdown(f"- **Title:** {meta.get('title', 'N/A')}")
                st.markdown(f"- **Author:** {meta.get('author', 'N/A')}")
                st.markdown(f"- **Size:** {meta.get('file_size_kb', 'N/A')} KB")

        if st.session_state.chunk_stats:
            stats = st.session_state.chunk_stats
            with st.expander("Chunk Statistics"):
                st.markdown(f"- **Total chunks:** {stats.get('count', 'N/A')}")
                st.markdown(f"- **Avg chunk size:** {stats.get('avg_chars', 'N/A')} chars")
                st.markdown(f"- **Min chunk size:** {stats.get('min_chars', 'N/A')} chars")
                st.markdown(f"- **Max chunk size:** {stats.get('max_chars', 'N/A')} chars")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            # Clear only the chat history, keep the index
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        with col2:
            # Wipe everything — delete index from disk and reset state
            if st.button("🔄 New PDF", use_container_width=True):
                delete_vector_store()
                st.session_state.chain         = None
                st.session_state.pdf_processed = False
                st.session_state.pdf_name      = None
                st.session_state.chunk_stats   = None
                st.session_state.pdf_metadata  = None
                st.session_state.messages      = []
                if "_pending_upload" in st.session_state:
                    del st.session_state["_pending_upload"]
                st.rerun()

    # --- Settings expander ---
    st.divider()
    with st.expander("⚙️ Model Settings"):
        chat_model = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
        embed_model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
        top_k = os.getenv("TOP_K_CHUNKS", "4")
        chunk_size = os.getenv("CHUNK_SIZE", "1000")
        chunk_overlap = os.getenv("CHUNK_OVERLAP", "200")

        st.markdown(f"**Chat model:** `{chat_model}`")
        st.markdown(f"**Embedding model:** `{embed_model}`")
        st.markdown(f"**Top-K chunks retrieved:** `{top_k}`")
        st.markdown(f"**Chunk size:** `{chunk_size}` chars")
        st.markdown(f"**Chunk overlap:** `{chunk_overlap}` chars")
        st.caption("Change these values in your `.env` file.")


# =============================================================================
# MAIN AREA
# =============================================================================

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>📄 PDF RAG Chatbot</h1>
    <p>Upload any PDF and ask questions about its contents using Gemini 2.5 Flash + semantic search</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# HANDLE PENDING PDF INGESTION
# When the user clicks "Process PDF" in the sidebar, we set a flag in
# session_state and call st.rerun(). On the next run, we detect the flag
# here and run the actual ingestion in the main content area (so the
# progress messages appear in the centre of the screen, not crammed in
# the sidebar).
# =============================================================================

if "_pending_upload" in st.session_state:
    pending = st.session_state["_pending_upload"]
    del st.session_state["_pending_upload"]  # clear the flag immediately

    st.subheader("🔄 Processing PDF...")
    st.caption(f"File: `{pending.name}`")
    st.divider()

    success = ingest_pdf(pending)

    if success:
        st.balloons()
        st.success("🎉 Your PDF is ready! Ask a question below.")
        st.rerun()  # re-run to show the clean chat UI
    else:
        st.warning("Ingestion failed. Please check the errors above and try again.")
    st.stop()  # don't render the chat UI during ingestion


# =============================================================================
# PRE-CHAT STATE: No PDF loaded yet
# =============================================================================

if not st.session_state.pdf_processed:
    # Show a friendly landing page with instructions
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("🚀 How to get started")
        st.markdown("""
        1. **Upload a PDF** using the sidebar on the left
        2. Click **Process PDF** — this will:
           - Extract all text from your PDF
           - Split it into smart overlapping chunks
           - Embed each chunk using OpenAI's embedding model
           - Store the vectors in a local FAISS index
        3. **Ask questions** in the chat box below
        4. The AI will find the most relevant parts of your document
           and answer based only on that content
        """)

    with col_right:
        st.subheader("⚡ What happens under the hood")
        st.markdown("""
        ```
        PDF
         ↓ PyMuPDF (text extraction)
        Raw Text
         ↓ RecursiveCharacterTextSplitter
        Chunks (1000 chars, 200 overlap)
         ↓ Google text-embedding-004
        Vectors [0.21, -0.83, 0.44, ...]
         ↓ FAISS vector store (saved to disk)

        ── At query time ──

        Your Question
         ↓ Embed question (same model)
        Query Vector
         ↓ FAISS similarity search → top 4 chunks
        Context
         ↓ ChatPromptTemplate + Gemini 2.5 Flash
        Answer (streamed token by token)
        ```
        """)

    if vector_store_exists():
        st.info(
            "💡 A previously processed PDF index was found on disk. "
            "You can load it from the sidebar without re-uploading the PDF.",
            icon="💾"
        )

    st.stop()  # Don't render the chat UI until a PDF is loaded


# =============================================================================
# CHAT INTERFACE
# Rendered only after a PDF has been successfully ingested.
# =============================================================================

st.subheader(f"💬 Chat with: `{st.session_state.pdf_name}`")
st.caption(
    f"Asking GPT about your document · "
    f"{st.session_state.chunk_stats.get('count', '?')} chunks indexed · "
    f"Top-{os.getenv('TOP_K_CHUNKS', '4')} retrieval"
)
st.divider()

# --- Render existing chat history ---
# We loop through session_state.messages and render each one.
# st.chat_message("user") and st.chat_message("assistant") render
# the appropriate avatar and bubble styling automatically.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show retrieved source chunks for assistant messages (if stored)
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📎 View retrieved source chunks", expanded=False):
                for i, chunk in enumerate(message["sources"], start=1):
                    source = chunk.get("source", "Document")
                    content = chunk.get("content", "")
                    st.markdown(
                        f'<div class="chunk-box">'
                        f'<strong>Chunk {i} | {source}</strong><br>{content[:500]}{"..." if len(content) > 500 else ""}'
                        f'</div>',
                        unsafe_allow_html=True
                    )


# --- Chat Input ---
# st.chat_input() renders a fixed input bar at the bottom of the page.
# It returns the submitted text (or None if nothing submitted this run).
# The walrus operator (:=) assigns AND checks in one line.

if question := st.chat_input(
    placeholder="Ask a question about your PDF...",
    key="chat_input",
):
    # Guard: make sure the chain is available
    if st.session_state.chain is None:
        st.error("❌ The chain is not initialised. Please reload the page and process a PDF.")
        st.stop()

    # --- Add user message to history and display it ---
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # --- Build chat history for the chain ---
    # We pass ALL previous messages to the chain so it has conversation memory.
    # trim_chat_history keeps only the last N turns to avoid context overflow.
    lc_history = trim_chat_history(
        build_chat_history(st.session_state.messages[:-1]),  # exclude the just-added question
        max_turns=10,
    )

    # --- Stream the AI response ---
    with st.chat_message("assistant"):
        # st.empty() creates a placeholder we can update token by token
        response_placeholder = st.empty()
        full_response = ""

        try:
            # get_answer_stream() yields string tokens from chain.stream()
            # We accumulate them and update the placeholder each time.
            # The "▌" is a blinking cursor indicator while streaming.
            for token in get_answer_stream(
                chain=st.session_state.chain,
                question=question,
                chat_history=lc_history,
            ):
                full_response += token
                response_placeholder.markdown(full_response + "▌")

            # Final render without cursor
            response_placeholder.markdown(full_response)

        except Exception as e:
            error_msg = f"❌ Error generating response: {str(e)}"
            response_placeholder.error(error_msg)
            full_response = error_msg

    # --- Save assistant response to history ---
    st.session_state.messages.append({
        "role":    "assistant",
        "content": full_response,
    })


# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.caption(
    "Built with [LangChain](https://python.langchain.com) · "
    "[FAISS](https://github.com/facebookresearch/faiss) · "
    "[Gemini 2.5 Flash](https://aistudio.google.com) · "
    "[Streamlit](https://streamlit.io)"
)
