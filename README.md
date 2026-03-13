# 📄 PDF RAG Chatbot

A fully local + cloud RAG (Retrieval-Augmented Generation) pipeline that lets you upload any PDF and chat with it using GPT.

---

## 📋 Table of Contents

- [What is RAG?](#what-is-rag)
- [The Full Pipeline](#the-full-pipeline)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the App](#running-the-app)
- [How Every File Works](#how-every-file-works)
  - [loader.py](#loaderpy)
  - [splitter.py](#splitterpy)
  - [embedder.py](#embedderpy)
  - [retriever.py](#retrieverpy)
  - [chain.py](#chainpy)
  - [app.py](#apppy)
- [What LangChain Is Doing](#what-langchain-is-doing)
- [Configuration](#configuration)
- [Cost Estimate](#cost-estimate)
- [Common Issues](#common-issues)

---

## What is RAG?

**RAG = Retrieval-Augmented Generation**

A plain LLM (like GPT-4) only knows what it was trained on — it has no idea what's in your PDF.
RAG solves this by:

1. **Ingestion phase** (done once per document):
   - Parse the PDF into raw text
   - Break that text into small overlapping chunks
   - Convert each chunk into a vector embedding (a list of numbers representing its meaning)
   - Store all vectors in a vector database (FAISS)

2. **Query phase** (done on every question):
   - Embed the user's question using the same model
   - Search the vector DB for the chunks most similar to the question
   - Inject those chunks into the LLM prompt as context
   - The LLM reads that context and answers accordingly

The LLM never "learns" from your PDF — it just reads the relevant excerpts in its prompt each time, like giving it a cheat sheet.

---

## The Full Pipeline

```
┌─────────────┐
│   PDF File  │
└──────┬──────┘
       │ 1. PyMuPDF (fitz) — extract raw text, page by page
       ▼
┌──────────────────────┐
│  Raw Text String     │  "  [Page 1]\nIntroduction...\n\n[Page 2]..."
└──────┬───────────────┘
       │ 2. RecursiveCharacterTextSplitter — break into chunks
       ▼
┌──────────────────────────────┐
│  List of Document chunks     │  [Document(page_content="...", metadata={...}), ...]
│  ~1000 chars each, 200 overlap│
└──────┬───────────────────────┘
       │ 3. OpenAIEmbeddings — call OpenAI Embeddings API
       ▼
┌──────────────────────────────────┐
│  List of float vectors           │  [[0.021, -0.834, 0.412, ...], ...]
│  1536 dimensions per chunk       │  (text-embedding-3-small)
└──────┬───────────────────────────┘
       │ 4. FAISS.from_documents() — build the index
       ▼
┌───────────────────────┐
│  FAISS Vector Index   │  stored in vector_store/index.faiss
│  (saved to disk)      │              vector_store/index.pkl
└───────────────────────┘


         ──── At Query Time ────


User types: "What is the refund policy?"
       │
       │ 5. Same embedding model embeds the question
       ▼
  Query vector: [0.019, -0.829, 0.408, ...]
       │
       │ 6. FAISS similarity search → top 4 closest chunks
       ▼
┌──────────────────────────────────┐
│  Top 4 most relevant chunks      │
│  (cosine similarity to question) │
└──────┬───────────────────────────┘
       │ 7. format_docs() joins chunks into one string
       ▼
┌──────────────────────────────────────────────────┐
│  ChatPromptTemplate fills in {context}           │
│  and {question} + injects {chat_history}         │
└──────┬───────────────────────────────────────────┘
       │ 8. ChatOpenAI → GPT API call (streamed)
       ▼
┌──────────────────┐
│  Answer tokens   │  streamed token by token to Streamlit UI
└──────────────────┘
```

---

## Project Structure

```
rag_pipeline_test/
│
├── .env                    ← Your API keys (copy from .env.example)
├── .env.example            ← Template showing available env vars
├── .gitignore              ← Ignores .env, vector_store/, uploads/, etc.
├── requirements.txt        ← All pip dependencies
├── app.py                  ← Streamlit UI — main entry point
│
├── rag/                    ← Core RAG logic (Python package)
│   ├── __init__.py         ← Makes rag/ importable as a package
│   ├── loader.py           ← PDF text extraction (PyMuPDF)
│   ├── splitter.py         ← Text chunking (LangChain splitter)
│   ├── embedder.py         ← Embeddings + FAISS store management
│   ├── retriever.py        ← Similarity search helpers
│   └── chain.py            ← LangChain LCEL chain + streaming
│
├── vector_store/           ← Auto-created. Holds FAISS index files.
│   ├── index.faiss         ← Binary float vectors
│   └── index.pkl           ← Document texts + metadata (pickle)
│
└── uploads/                ← Auto-created. Uploaded PDFs saved here.
```

---

## Setup & Installation

### 1. Clone / open the project

```bash
cd rag_pipeline_test
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create your `.env` file

```bash
# Copy the template
cp .env.example .env
```

Then open `.env` and fill in your OpenAI API key:

```
OPENAI_API_KEY=sk-your-actual-key-here
```

Get a key at: https://platform.openai.com/api-keys

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Running the App

1. Open `http://localhost:8501` in your browser
2. Upload a PDF using the sidebar
3. Click **Process PDF** — watch the 4-step ingestion pipeline run
4. Type a question in the chat box
5. The answer streams back in real time

**Tip:** After processing a PDF once, the FAISS index is saved to disk.
Next time you start the app, click **Load Previous Index** to skip re-embedding.

---

## How Every File Works

---

### `loader.py`

**Tool used:** `PyMuPDF` (imported as `fitz`)

PyMuPDF is one of the fastest and most accurate PDF text extraction libraries.
It opens the PDF, iterates over every page, and calls `page.get_text("text")`
to pull out plain text.

```python
doc = fitz.open(file_path)
for page_num in range(doc.page_count):
    page = doc[page_num]
    page_text = page.get_text("text")
```

**Key functions:**
- `load_pdf(path)` — returns one giant string with `[Page N]` markers
- `get_pdf_metadata(path)` — returns title, author, page count, file size

**Limitation:** Only works on text-based PDFs. Scanned PDFs (pure images)
won't extract any text without an OCR step (e.g., `pytesseract`).

---

### `splitter.py`

**Tool used:** `langchain_text_splitters.RecursiveCharacterTextSplitter`

This is one of the most important pieces of the pipeline.
We can't embed the whole document as one chunk — it would be too large
and every question would retrieve the whole thing (useless).

**Why "Recursive"?**
The splitter tries separators in order of preference:
1. `"\n\n"` — paragraph breaks (best: preserves semantic units)
2. `"\n"` — line breaks
3. `". "` — sentence boundaries
4. `" "` — word boundaries
5. `""` — individual characters (last resort)

It keeps trying smaller separators until every chunk fits within `chunk_size`.

**Why overlap?**
If a key sentence happens to fall at the boundary between two chunks,
without overlap it would be split in half and lose context.
With `chunk_overlap=200`, the last 200 characters of each chunk are
repeated at the start of the next one.

```
Chunk 1: "...The contract expires on December 31st, 2024. Renewal requires"
Chunk 2: "requires written notice 30 days in advance..."
                 ↑ this bit is the overlap
```

**Output:** `List[Document]` — each `Document` has:
- `.page_content` — the chunk text
- `.metadata` — dict (we store `{"source": "filename.pdf"}`)

---

### `embedder.py`

**Tools used:** `langchain_openai.OpenAIEmbeddings`, `langchain_community.vectorstores.FAISS`

This file has two responsibilities: creating embeddings and managing FAISS.

#### What is an Embedding?

An embedding is a list of floating-point numbers that encodes the **semantic meaning** of text.

```
"The cat sat on the mat"       → [0.021, -0.834, 0.412, 0.003, ...]  (1536 numbers)
"A feline rested on the rug"   → [0.019, -0.829, 0.408, 0.001, ...]  (1536 numbers)
"The stock market crashed"     → [0.721,  0.334, -0.91, 0.552, ...]  (very different)
```

Texts with similar meanings have vectors that point in similar directions in 1536-dimensional space.

#### What is FAISS?

FAISS (Facebook AI Similarity Search) is a C++ library (with Python bindings) that:
- Stores all your vectors in memory efficiently
- Can find the nearest neighbours among millions of vectors in milliseconds
- Saves the index to disk as two files: `index.faiss` (binary vectors) + `index.pkl` (document texts)

`FAISS.from_documents(documents, embeddings)` does this in one call:
1. Extracts `.page_content` from each Document
2. Calls OpenAI Embeddings API in batches
3. Builds the FAISS index from the returned vectors
4. Stores the mapping: vector ID → Document object

**Why `allow_dangerous_deserialization=True` when loading?**
FAISS uses Python's `pickle` format to store document texts (`.pkl` file).
Pickle can run arbitrary code when loaded, so LangChain v0.2+ requires you
to explicitly acknowledge this risk. Since we wrote the file ourselves during
ingestion, it's safe.

**Key functions:**
- `create_vector_store(documents)` — embed all chunks, build FAISS, save to disk
- `load_vector_store()` — restore a saved FAISS index from disk
- `vector_store_exists()` — check if a saved index exists
- `add_documents_to_store(new_docs)` — add more docs to an existing index (multi-PDF)
- `delete_vector_store()` — wipe the saved index

---

### `retriever.py`

**Tool used:** `FAISS.as_retriever()` / `similarity_search_with_score()`

This file handles the query-time retrieval: given a question, find the most relevant chunks.

**How similarity search works:**
1. The question is embedded using the same model used during ingestion (critical!)
2. FAISS computes the distance between the query vector and every stored vector
3. The top-K closest vectors (= most semantically similar chunks) are returned

**Why not keyword search (like grep)?**
Keyword search only finds exact word matches. Semantic search understands meaning:

| Query | Keyword search finds | Semantic search finds |
|---|---|---|
| "late payment penalty" | only chunks with those exact words | also chunks about "overdue fees", "past due charges", etc. |
| "how do I cancel?" | only "cancel" | also "termination", "discontinue service", "unsubscribe" |

**Key functions:**
- `get_relevant_chunks(query, store, k=4)` — returns top-k Documents
- `get_relevant_chunks_with_scores(query, store, k=4)` — returns (Document, score) tuples
- `get_relevant_chunks_filtered(query, store, score_threshold=1.0)` — filters out low-relevance chunks
- `format_retrieved_chunks(documents)` — formats chunks into a readable string

---

### `chain.py`

**Tools used:** `ChatOpenAI`, `ChatPromptTemplate`, `StrOutputParser`, `RunnablePassthrough`, `RunnableLambda`

This is the brain of the pipeline — it wires everything together into a single callable chain.

#### What is LCEL (LangChain Expression Language)?

LCEL is LangChain's modern way to compose pipelines using the `|` (pipe) operator.
Every component is a "Runnable" — an object with `.invoke()`, `.stream()`, and `.batch()` methods.

```
chain = component_A | component_B | component_C
```

The output of A is the input of B, the output of B is the input of C.
Just like Unix pipes: `cat file.txt | grep "error" | wc -l`

#### The Full Chain

```python
chain = (
    {
        "context":      RunnableLambda(lambda x: x["question"]) | retriever | format_docs,
        "question":     RunnableLambda(lambda x: x["question"]),
        "chat_history": RunnableLambda(lambda x: x.get("chat_history", [])),
    }
    | PROMPT
    | llm
    | StrOutputParser()
)
```

**Step by step:**

| Step | Component | Input | Output |
|---|---|---|---|
| 1 | `RunnableLambda` | `{"question": "...", "chat_history": [...]}` | `"..."` (just the question string) |
| 2 | `retriever` | `"What is the refund policy?"` | `[Document, Document, Document, Document]` |
| 3 | `format_docs` | `List[Document]` | `"[Excerpt 1]...\n\n---\n\n[Excerpt 2]..."` |
| 4 | `PROMPT` | `{"context": "...", "question": "...", "chat_history": [...]}` | `[SystemMessage, HumanMessage]` |
| 5 | `ChatOpenAI` | `[SystemMessage, HumanMessage]` | Stream of `AIMessageChunk` objects |
| 6 | `StrOutputParser` | `AIMessageChunk` | Plain `str` tokens |

#### What is `RunnablePassthrough`?

A no-op Runnable — it passes its input straight through unchanged.
It's used when you want a value to flow to the next step without any transformation.

#### What is `MessagesPlaceholder`?

A special slot in the prompt template that accepts a list of LangChain message objects
(`HumanMessage`, `AIMessage`). This is how we inject conversation history into the prompt,
giving the LLM memory of what was said earlier in the session.

#### What is `StrOutputParser`?

The LLM returns `AIMessage` objects (or `AIMessageChunk` when streaming).
`StrOutputParser` extracts just the `.content` string from each one.
Without it, you'd get objects like `AIMessageChunk(content="The", ...)` instead of `"The"`.

#### Streaming

`chain.stream(input)` returns a Python generator that yields string tokens one by one:

```python
for token in chain.stream({"question": "...", "chat_history": [...]}):
    print(token, end="", flush=True)
    # prints: "The" " refund" " policy" " states" " that" ...
```

Streamlit's `st.empty()` placeholder is updated on every token, creating the typewriter effect.

#### Conversation Memory

The chain doesn't store state internally — it's stateless.
We manage history externally in `st.session_state.messages` (a list of dicts).
Before each call, `build_chat_history()` converts that list into LangChain message objects
and `trim_chat_history()` keeps only the last 10 turns to avoid hitting token limits.

---

### `app.py`

**Tool used:** `streamlit`

The Streamlit frontend. Here's what makes it tick:

#### How Streamlit Works

Streamlit re-runs the **entire script** from top to bottom on every user interaction.
This means all local variables are destroyed and recreated on every click or keypress.

To persist data between re-runs, we use `st.session_state` — a dict-like object that
survives the entire browser session:

```python
if "messages" not in st.session_state:
    st.session_state.messages = []  # initialised once, persists forever
```

**What we persist:**
| Key | Type | Purpose |
|---|---|---|
| `messages` | `List[dict]` | Full chat history `[{"role": "user", "content": "..."}]` |
| `chain` | LangChain Runnable | The built LCEL chain (expensive to rebuild each run) |
| `pdf_processed` | `bool` | Whether a PDF has been ingested |
| `pdf_name` | `str` | Display name of the loaded PDF |
| `chunk_stats` | `dict` | Stats for the sidebar display |
| `pdf_metadata` | `dict` | PDF title, author, page count |

#### The Pending Upload Pattern

Streamlit can't easily run a long process (ingestion) from inside the sidebar
and display progress in the main area. We solve this with a flag:

1. User clicks "Process PDF" in sidebar → we set `st.session_state["_pending_upload"] = file` and call `st.rerun()`
2. On the next run, the main area detects the flag, deletes it, and runs `ingest_pdf()`
3. Progress messages render in the centre of the screen
4. On success, `st.rerun()` again to show the clean chat UI

#### Streaming in Streamlit

```python
response_placeholder = st.empty()  # a blank container we can update
full_response = ""

for token in get_answer_stream(chain, question, lc_history):
    full_response += token
    response_placeholder.markdown(full_response + "▌")  # ▌ = fake cursor

response_placeholder.markdown(full_response)  # final render, no cursor
```

---

## What LangChain Is Doing

LangChain is a framework that provides:

1. **Abstractions** — standard interfaces for LLMs, embeddings, vector stores, retrievers, etc.
   You can swap OpenAI for Anthropic or FAISS for Pinecone without rewriting your pipeline logic.

2. **LCEL (LangChain Expression Language)** — the `|` pipe syntax for composing pipelines
   in a clean, readable way that supports streaming, batching, and async out of the box.

3. **Integrations** — pre-built connectors to 100+ LLM providers, vector stores, document loaders,
   and more, all with a consistent API.

4. **Prompt templates** — `ChatPromptTemplate` handles filling in variables, formatting messages
   for chat vs. completion models, and injecting structured data like chat history.

**What LangChain is NOT doing here:**
- It's not magically making the LLM smarter
- It's not storing your data in the cloud
- It's not making API calls you didn't ask for

Every API call that happens is either to OpenAI Embeddings (during ingestion)
or to OpenAI Chat Completions (during each question). That's it.

---

## Configuration

All settings are controlled via the `.env` file:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | (required) | Your OpenAI API key |
| `CHAT_MODEL` | `gpt-4o-mini` | GPT model for answering questions |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI model for embeddings |
| `CHUNK_SIZE` | `1000` | Max characters per chunk |
| `CHUNK_OVERLAP` | `200` | Character overlap between chunks |
| `TOP_K_CHUNKS` | `4` | Number of chunks to retrieve per question |

**Important:** If you change `EMBEDDING_MODEL` after already ingesting a PDF,
delete `vector_store/` and re-process the PDF. The query embeddings must use
the same model as the index embeddings or similarity search will return garbage.

---

## Cost Estimate

Using the defaults (`text-embedding-3-small` + `gpt-4o-mini`):

| Operation | Cost |
|---|---|
| Embedding a 10-page PDF | ~$0.0001 (fractions of a cent) |
| Embedding a 100-page PDF | ~$0.001 (less than 0.1 cent) |
| Each question (4 chunks context + answer) | ~$0.0002–$0.001 depending on answer length |
| 100 questions on a 100-page PDF | ~$0.05–$0.10 |

The embedding cost is one-time per document (saved to disk). Questions are where ongoing cost accumulates.

---

## Common Issues

**"No text could be extracted from the PDF"**
→ Your PDF is likely scanned (image-based). Text extraction only works on text-layer PDFs.
→ Solution: Use an OCR tool like `pytesseract` + `pdf2image` as a pre-processing step.

**"OpenAI API key missing"**
→ Make sure you created a `.env` file (not just `.env.example`) with your actual key.
→ Make sure `load_dotenv()` is called before any OpenAI imports.

**Answers are wrong or irrelevant**
→ Try lowering `CHUNK_SIZE` (e.g., 500) for more precise retrieval.
→ Try increasing `TOP_K_CHUNKS` (e.g., 6) to give the LLM more context.
→ The PDF may have complex formatting (tables, columns) that doesn't extract cleanly.

**"allow_dangerous_deserialization" error**
→ You're loading a FAISS index without the flag. This is handled in `load_vector_store()`.
→ Only occurs if you're calling `FAISS.load_local()` directly somewhere.

**Streamlit reruns losing state**
→ Make sure you're storing everything important in `st.session_state`.
→ Never rely on module-level global variables in Streamlit — they reset on re-run.