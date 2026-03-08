import os
import sys
import tempfile

import streamlit as st

# Add src to path so imports work when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from ingest import ingest_documents, get_vectorstore
from chain import create_rag_chain, ask_question
from config import UPLOAD_DIR, USE_LOCAL_MODEL

# --- Page Config ---
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📄",
    layout="wide",
)

st.title("📄 RAG Chatbot — Chat with Your Documents")
st.caption("Upload a PDF and ask questions about its content")

# --- Sidebar ---
with st.sidebar:
    st.header("📁 Upload Documents")

    mode = "🏠 Local (Ollama)" if USE_LOCAL_MODEL else "☁️ OpenAI"
    st.info(f"Running in **{mode}** mode")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Processing document... This may take a moment."):
            vectorstore = ingest_documents(file_path)
            st.session_state["vectorstore"] = vectorstore
            st.session_state["chain"] = create_rag_chain(vectorstore)

        st.success(f"✅ '{uploaded_file.name}' processed successfully!")

    if st.button("🗑️ Clear Chat History"):
        st.session_state["messages"] = []
        if "chain" in st.session_state:
            del st.session_state["chain"]
        st.rerun()

    st.divider()
    st.markdown(
        "**How it works:**\n"
        "1. Upload a PDF document\n"
        "2. The document is split into chunks and embedded\n"
        "3. Ask questions and get answers with citations\n"
    )

# --- Load existing vectorstore on startup ---
if "vectorstore" not in st.session_state:
    existing_store = get_vectorstore()
    if existing_store:
        st.session_state["vectorstore"] = existing_store
        st.session_state["chain"] = create_rag_chain(existing_store)

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📎 Sources"):
                for src in message["sources"]:
                    st.markdown(
                        f"- **Page {src['page']}** from `{os.path.basename(src['source'])}`: "
                        f"_{src['content_preview']}_"
                    )

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Show user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    if "chain" not in st.session_state:
        with st.chat_message("assistant"):
            st.warning("⚠️ Please upload a PDF document first!")
        st.session_state["messages"].append({
            "role": "assistant",
            "content": "⚠️ Please upload a PDF document first!",
        })
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask_question(st.session_state["chain"], prompt)

            st.markdown(result["answer"])

            if result["sources"]:
                with st.expander("📎 Sources"):
                    for src in result["sources"]:
                        st.markdown(
                            f"- **Page {src['page']}** from `{os.path.basename(src['source'])}`: "
                            f"_{src['content_preview']}_"
                        )

        st.session_state["messages"].append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })
