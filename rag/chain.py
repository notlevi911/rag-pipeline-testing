# =============================================================================
# rag/chain.py
# =============================================================================
# WHAT THIS FILE DOES:
#   Wires together the retriever + prompt template + LLM into a single
#   "chain" that you can call with a question and get a streamed answer.
#
# WHAT IS LCEL (LangChain Expression Language)?
#   LCEL is LangChain's modern way to compose pipelines using the pipe
#   operator (|), similar to Unix pipes. Each component is a "Runnable"
#   that receives an input, does something, and passes output to the next.
#
#   Example chain flow:
#     question (str)
#       → retriever         → finds relevant Document chunks from FAISS
#       → format_docs()     → joins chunks into one context string
#       → prompt template   → fills in {context} and {question} placeholders
#       → ChatGoogleGenerativeAI → sends the filled prompt to Gemini, gets tokens
#       → StrOutputParser   → converts the AIMessage object to a plain string
#
#   Written with LCEL it looks like:
#     chain = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )
#
# WHAT IS RunnablePassthrough?
#   It's a no-op Runnable — it just passes its input straight through unchanged.
#   We use it for the "question" key so the original question string reaches
#   the prompt template without modification, while the "context" key goes
#   through the retriever pipeline separately.
#
# WHAT IS ChatPromptTemplate?
#   A template with placeholders ({context}, {question}) that gets filled in
#   at runtime. It formats the messages into the correct shape for a chat LLM:
#     [SystemMessage("You are helpful...context here..."),
#      HumanMessage("What is the refund policy?")]
#
# STREAMING:
#   Instead of waiting for the full response, chain.stream() yields tokens
#   one by one as Gemini generates them. This makes the UI feel responsive.
#
# CONVERSATION HISTORY:
#   The basic chain below is STATELESS — each question is independent.
#   We pass the full chat history manually via the `chat_history` parameter
#   and inject it into the prompt so the LLM has conversation context.
# =============================================================================

import os
from typing import Generator, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.vectorstores import FAISS


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================
# This is the instruction set we give the LLM on every call.
# It has three parts:
#   1. A system message — defines the LLM's role, rules, and injects context
#   2. A chat history placeholder — injects previous turns so LLM has memory
#   3. A human message — the user's current question
#
# {context}  → filled with the retrieved document chunks (from FAISS)
# {question} → filled with the user's current question
# MessagesPlaceholder → filled with the list of previous chat messages
#
# IMPORTANT: The context is injected into the SYSTEM message, not the human
# message. This is intentional — system-level context is given higher
# "authority" by the model than user-level context.

SYSTEM_TEMPLATE = """\
You are a helpful, precise document assistant. Your job is to answer the user's \
questions strictly based on the provided document context below.

RULES:
- Answer ONLY using the context provided. Do not use outside knowledge.
- If the answer is not found in the context, respond with:
  "I couldn't find that information in the uploaded document."
- Be concise, clear, and well-structured.
- If helpful, quote directly from the document (use quotation marks).
- If the user asks a follow-up question, use the conversation history to understand what they're referring to.

---
DOCUMENT CONTEXT:
{context}
---
"""

# Build the ChatPromptTemplate with:
#   - A system message (rules + context)
#   - A placeholder for chat history (list of HumanMessage/AIMessage objects)
#   - A human message (the current question)
PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    MessagesPlaceholder(variable_name="chat_history"),  # injected conversation history
    ("human", "{question}"),
])


# =============================================================================
# HELPER: Format retrieved docs into a string
# =============================================================================

def format_docs(docs) -> str:
    """
    Joins a list of retrieved Document chunks into a single formatted string.

    This string gets inserted into the {context} slot of the prompt template.
    Each chunk is separated by a "---" divider so the LLM can distinguish
    between different sections of the document.

    Args:
        docs (List[Document]): Retrieved Document objects from FAISS.

    Returns:
        str: A single string of all chunk texts, ready to inject into the prompt.
             Returns a fallback string if no docs were found.
    """
    if not docs:
        return "No relevant context was found in the document."

    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Document")
        parts.append(f"[Excerpt {i} | {source}]\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(parts)


# =============================================================================
# CHAIN BUILDER
# =============================================================================

def build_chain(vector_store: FAISS, model: Optional[str] = None):
    """
    Constructs and returns a LangChain LCEL chain for RAG question answering
    powered by Google Gemini 2.5 Flash.

    THE CHAIN STEP BY STEP:
    -----------------------
    Input to the chain is a dict:
        {
            "question":     "What is the refund policy?",
            "chat_history": [HumanMessage(...), AIMessage(...), ...]
        }

    Step 1 — Context retrieval:
        The "question" string is passed to the FAISS retriever.
        The retriever embeds the question with Google's embedding model and
        finds the top-K matching chunks by cosine similarity.
        Those chunks are passed through format_docs() → a single context string.

    Step 2 — Question & history passthrough:
        RunnableLambda extracts the "question" and "chat_history" values
        from the input dict and passes them through unchanged.

    Step 3 — Prompt formatting:
        The PROMPT template fills in {context}, {question}, and {chat_history}.
        Output: a list of BaseMessage objects (SystemMessage + HumanMessage).

    Step 4 — LLM call:
        ChatGoogleGenerativeAI sends the formatted messages to the Gemini API.
        With streaming=True, it yields AIMessageChunk objects token by token.

    Step 5 — Output parsing:
        StrOutputParser extracts the .content string from each AIMessageChunk.
        Final output: a stream of plain strings (tokens).

    WHY gemini-2.5-flash?
        - It's FREE under Google AI Studio's free tier (no billing needed)
        - It's fast and has a 1M token context window
        - It's a "thinking" model — great at reasoning over document content
        - Supports streaming responses

    WHY temperature=0?
        Temperature controls randomness. 0 = always picks the most likely token
        = deterministic, factual answers. Good for document Q&A where we want
        accurate retrieval, not creative generation.

    Args:
        vector_store (FAISS):   A loaded FAISS vector store to retrieve from.
        model (str, optional):  Gemini model name override. Defaults to env var
                                CHAT_MODEL or "gemini-2.5-flash".

    Returns:
        A LangChain Runnable chain that accepts:
            {"question": str, "chat_history": List[BaseMessage]}
        and returns a streamable/invokable string output.
    """
    chat_model = model or os.getenv("CHAT_MODEL", "gemini-2.5-flash")

    # --- LLM ---
    # ChatGoogleGenerativeAI wraps Google's Gemini chat API.
    # It reads GOOGLE_API_KEY automatically from the environment.
    # streaming=True enables token-by-token streaming via .stream()
    # temperature=0 makes answers deterministic and factual
    llm = ChatGoogleGenerativeAI(
        model=chat_model,
        temperature=0,
        streaming=True,
        # google_api_key is auto-read from GOOGLE_API_KEY env var
    )

    # --- Retriever ---
    # .as_retriever() converts the FAISS store into a LangChain Retriever object.
    # A Retriever is a Runnable that accepts a string and returns List[Document].
    # search_type="similarity" uses cosine similarity.
    # k=4 means: return the 4 most relevant chunks for each question.
    k = int(os.getenv("TOP_K_CHUNKS", "4"))
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    # --- LCEL Chain ---
    #
    # The chain is built using Python's | (pipe) operator.
    # Each component must be a "Runnable" (has .invoke(), .stream(), .batch()).
    #
    # The first element is a dict of Runnables — LangChain runs them in parallel
    # and merges the results into a single dict before passing to the next step:
    #
    #   {
    #     "context":      retriever | format_docs   ← runs on "question" input
    #     "question":     RunnableLambda(...)        ← extracts "question" key
    #     "chat_history": RunnableLambda(...)        ← extracts "chat_history" key
    #   }
    #
    # Note: When the input is a dict, each RunnableLambda pulls the
    # matching key from the input dict. So "context" gets built from
    # input["question"], and "question" + "chat_history" are forwarded as-is.

    chain = (
        {
            # retrieve context using the question, then format into a string
            "context": (
                RunnableLambda(lambda x: x["question"])  # extract question from input dict
                | retriever                               # embed question → find top-k chunks
                | format_docs                            # join chunks → single context string
            ),
            # pass the question string through unchanged to the prompt
            "question": RunnableLambda(lambda x: x["question"]),

            # pass the chat history list through unchanged to the prompt
            "chat_history": RunnableLambda(lambda x: x.get("chat_history", [])),
        }
        | PROMPT            # fill in {context}, {question}, {chat_history} → list of messages
        | llm               # send messages to Gemini → stream of AIMessageChunk tokens
        | StrOutputParser()  # extract .content string from each chunk → stream of str tokens
    )

    return chain


# =============================================================================
# ANSWER FUNCTIONS
# =============================================================================

def get_answer_stream(
    chain,
    question: str,
    chat_history: Optional[List[BaseMessage]] = None,
) -> Generator[str, None, None]:
    """
    Streams the LLM's answer token by token.

    HOW STREAMING WORKS WITH LCEL:
    --------------------------------
    chain.stream(input) returns a generator. As Gemini generates each token,
    it's yielded immediately. The Streamlit app accumulates tokens and
    updates the UI in real-time, giving a "typewriter" effect.

    Without streaming (chain.invoke), you'd wait several seconds for the full
    response before anything appears in the UI. Streaming feels much faster.

    Args:
        chain:                    The built LCEL chain from build_chain().
        question (str):           The user's current question.
        chat_history (optional):  List of previous HumanMessage/AIMessage objects.
                                  Pass None or [] for a fresh conversation.

    Yields:
        str: Individual token strings as they arrive from the API.
    """
    input_payload = {
        "question": question,
        "chat_history": chat_history or [],
    }

    # .stream() returns a generator of string tokens (after StrOutputParser)
    for token in chain.stream(input_payload):
        yield token


def get_answer_full(
    chain,
    question: str,
    chat_history: Optional[List[BaseMessage]] = None,
) -> str:
    """
    Gets the full answer in one shot (no streaming).
    Useful for testing, scripting, or non-UI use cases.

    Args:
        chain:                   The built LCEL chain.
        question (str):          The user's question.
        chat_history (optional): Previous conversation turns.

    Returns:
        str: The complete answer as a single string.
    """
    input_payload = {
        "question": question,
        "chat_history": chat_history or [],
    }

    return chain.invoke(input_payload)


# =============================================================================
# CHAT HISTORY HELPERS
# =============================================================================

def build_chat_history(
    messages: List[dict],
) -> List[BaseMessage]:
    """
    Converts the Streamlit session_state messages list (list of dicts)
    into LangChain message objects that the prompt template understands.

    Streamlit stores messages as:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

    LangChain expects:
        [HumanMessage(content="..."), AIMessage(content="...")]

    Args:
        messages (List[dict]): Chat history from st.session_state.messages.
                               Each dict must have "role" and "content" keys.

    Returns:
        List[BaseMessage]: Converted list of LangChain message objects.
                           Skips any messages with unrecognised roles.
    """
    history: List[BaseMessage] = []

    for msg in messages:
        role    = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            history.append(HumanMessage(content=content))
        elif role == "assistant":
            history.append(AIMessage(content=content))
        # ignore system messages or anything else

    return history


def trim_chat_history(
    history: List[BaseMessage],
    max_turns: int = 10,
) -> List[BaseMessage]:
    """
    Trims the chat history to the last N turns to avoid exceeding the
    LLM's context window. Each "turn" = 1 human message + 1 AI message = 2 messages.

    WHY TRIM?
        Gemini 2.5 Flash has a 1M token context window — enormous.
        For most use cases you'll never hit the limit, but trimming is
        still good practice to keep prompts lean and responses fast.
        Keeping the last 10 turns (20 messages) is a safe, sensible default.

    Args:
        history (List[BaseMessage]): Full chat history.
        max_turns (int):             Max number of back-and-forth turns to keep.
                                     Each turn = 2 messages (human + AI).

    Returns:
        List[BaseMessage]: Trimmed history (most recent messages kept).
    """
    max_messages = max_turns * 2  # each turn is 2 messages

    if len(history) <= max_messages:
        return history

    # Keep only the most recent messages
    return history[-max_messages:]
