import os
import tempfile
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings

import google.generativeai as genai

# -------------------------
# LOAD ENV
# -------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY")
    st.stop()

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Multilingual RAG Pro", layout="wide")
st.title("🌍 Multilingual RAG (Perfect Version)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# -------------------------
# LLM
# -------------------------
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY
)

# -------------------------
# SAFE EMBEDDINGS (NO TORCH)
# -------------------------
embeddings = FakeEmbeddings(size=384)

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("Upload PDFs")

files = st.sidebar.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

# -------------------------
# PROCESS PDFs
# -------------------------
def process_files(uploaded_files):

    docs = []

    for file in uploaded_files:

        if file.name in st.session_state.processed_files:
            continue

        st.session_state.processed_files.add(file.name)

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            path = tmp.name

        loader = PyPDFLoader(path)
        pages = loader.load()

        for p in pages:
            p.metadata["source"] = file.name

        docs.extend(pages)

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(docs)

# -------------------------
# BUILD VECTOR DB
# -------------------------
if files:

    chunks = process_files(files)

    if chunks:

        if st.session_state.vector_db is None:
            st.session_state.vector_db = FAISS.from_documents(
                chunks,
                embeddings
            )
        else:
            st.session_state.vector_db.add_documents(chunks)

retriever = None

if st.session_state.vector_db:
    retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})

# -------------------------
# CHAT HISTORY
# -------------------------
for role, msg in st.session_state.chat_history:
    st.chat_message(role).write(msg)

# -------------------------
# 🔥 MULTILINGUAL QUERY REWRITER
# -------------------------
def rewrite_to_english(query):
    prompt = f"""
You are a query rewriting system.

Task:
Convert the user question into a clear English search query.

Rules:
- Preserve meaning
- Do NOT answer
- Only rewrite
- Remove language noise
- Make it good for document search

User query:
{query}
"""
    return llm.invoke(prompt).content.strip()

# -------------------------
# CHAT INPUT
# -------------------------
q = st.chat_input("Ask anything (any language)")

if q:

    st.chat_message("user").write(q)
    st.session_state.chat_history.append(("user", q))

    # STEP 1: rewrite query (IMPORTANT)
    query_en = rewrite_to_english(q)

    # STEP 2: retrieve context
    context = ""

    if retriever:
        docs = retriever.invoke(query_en)
        context = "\n\n".join([d.page_content for d in docs])

    # STEP 3: final answer (ENGLISH ONLY)
    prompt = f"""
You are a smart AI assistant.

RULES:
- Understand user in ANY language
- Always respond ONLY in English
- Use context if available
- Be clear and helpful

Context:
{context}

Question:
{query_en}

Answer:
"""

    ans = llm.invoke(prompt).content.strip()

    # STEP 4: output
    st.chat_message("assistant").write(ans)
    st.session_state.chat_history.append(("assistant", ans))