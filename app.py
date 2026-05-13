import os
import tempfile
import streamlit as st

from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings

# -------------------------
# API KEY (CLOUD SAFE)
# -------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)

if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY in Streamlit Secrets")
    st.stop()

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Multilingual RAG Pro", layout="wide")
st.title("🌍 Multilingual RAG (Perfect Stable Version)")

# -------------------------
# SESSION STATE
# -------------------------
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
# EMBEDDINGS (NO TORCH)
# -------------------------
embeddings = FakeEmbeddings(size=384)

# -------------------------
# SIDEBAR UPLOAD
# -------------------------
st.sidebar.title("📂 Upload PDFs")

files = st.sidebar.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

# -------------------------
# PROCESS PDFS
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

    chunks = splitter.split_documents(docs)

    return FAISS.from_documents(chunks, embeddings)

# -------------------------
# BUILD VECTOR DB
# -------------------------
if files:
    db = process_files(files)

    if db:
        if st.session_state.vector_db is None:
            st.session_state.vector_db = db
        else:
            st.session_state.vector_db.add_documents(db.docstore._dict.values())

# -------------------------
# RETRIEVER
# -------------------------
retriever = None

if st.session_state.vector_db:
    retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})

# -------------------------
# CHAT HISTORY
# -------------------------
for role, msg in st.session_state.chat_history:
    st.chat_message(role).write(msg)

# -------------------------
# MULTILINGUAL PROMPT ENGINE
# -------------------------
def build_prompt(user_query, context):
    return f"""
You are a multilingual intelligent assistant.

RULES:
- Understand ANY language (Telugu, Hindi, English, mixed)
- Always respond ONLY in English
- Do NOT translate output back
- Use context if available
- Be accurate and helpful

Context:
{context}

User Question:
{user_query}

Answer:
"""

# -------------------------
# CHAT INPUT
# -------------------------
q = st.chat_input("Ask anything in any language...")

if q:

    st.chat_message("user").write(q)
    st.session_state.chat_history.append(("user", q))

    # STEP 1: direct query (NO translation)
    query = q

    # STEP 2: retrieve context
    context = ""

    if retriever:
        docs = retriever.invoke(query)
        context = "\n\n".join([d.page_content for d in docs])

    # STEP 3: generate answer
    prompt = build_prompt(query, context)

    ans = llm.invoke(prompt).content.strip()

    # STEP 4: output
    st.chat_message("assistant").write(ans)
    st.session_state.chat_history.append(("assistant", ans))