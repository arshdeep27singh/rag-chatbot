import os
import sys
import tempfile

import streamlit as st

# Add src to path so imports work when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from ingest import ingest_documents, get_vectorstore
from chain import create_rag_chain, ask_question
from config import UPLOAD_DIR, LLM_PROVIDER

# --- Page Config ---
st.set_page_config(
    page_title="DocChat AI",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Global font */
    html, body, [class*="st-"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* Hero section */
    .hero {
        text-align: center;
        padding: 2rem 1rem 1rem 1rem;
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #6b7280;
        font-size: 1.15rem;
        margin-top: 0;
    }

    /* Feature cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.2rem;
        padding: 0.5rem 0 2rem 0;
    }
    .feature-card {
        background: #f8f9fc;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.2s ease;
    }
    .feature-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 16px rgba(102,126,234,0.12);
        transform: translateY(-2px);
    }
    .feature-icon {
        font-size: 2.4rem;
        margin-bottom: 0.5rem;
    }
    .feature-card h3 {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1f2937;
        margin: 0.3rem 0;
    }
    .feature-card p {
        font-size: 0.85rem;
        color: #6b7280;
        margin: 0;
    }

    /* Upload area */
    .upload-zone {
        background: linear-gradient(135deg, #f0f4ff 0%, #faf5ff 100%);
        border: 2px dashed #667eea;
        border-radius: 20px;
        padding: 2.5rem 1rem;
        text-align: center;
        margin: 1rem auto;
        max-width: 600px;
    }
    .upload-zone h3 {
        color: #667eea;
        font-weight: 700;
    }
    .upload-zone p {
        color: #6b7280;
        font-size: 0.9rem;
    }

    /* Status badge */
    .mode-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.5rem 0 1.5rem 0;
    }
    .mode-local {
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .mode-cloud {
        background: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
    }

    /* Chat container */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }

    /* Source cards */
    .source-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.85rem;
    }
    .source-card strong {
        color: #667eea;
    }

    /* Divider */
    .section-divider {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 1.5rem 0;
    }

    /* Compact file uploader */
    [data-testid="stFileUploader"] {
        max-width: 600px;
        margin: 0 auto;
    }

    /* Chat input styling */
    [data-testid="stChatInput"] {
        max-width: 800px;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)


# --- Session State Init ---
if "page" not in st.session_state:
    st.session_state["page"] = "home"
if "messages" not in st.session_state:
    st.session_state["messages"] = []


def go_to_chat():
    st.session_state["page"] = "chat"


def go_to_home():
    st.session_state["page"] = "home"


# --- Load existing vectorstore on startup ---
if "vectorstore" not in st.session_state:
    existing_store = get_vectorstore()
    if existing_store:
        st.session_state["vectorstore"] = existing_store
        st.session_state["chain"] = create_rag_chain(existing_store)


# ============================================================
# HOME PAGE
# ============================================================
if st.session_state["page"] == "home":

    # Hero
    st.markdown("""
    <div class="hero">
        <h1>💬 DocChat AI</h1>
        <p>Upload your documents and get instant, intelligent answers — powered by AI</p>
    </div>
    """, unsafe_allow_html=True)

    # Mode badge
    _badges = {
        "ollama": ("mode-local", "🏠 Running Locally with Ollama"),
        "groq": ("mode-cloud", "⚡ Running with Groq (Free)"),
    }
    _cls, _label = _badges.get(LLM_PROVIDER, _badges["groq"])
    st.markdown(f'<div style="text-align:center"><span class="mode-badge {_cls}">{_label}</span></div>', unsafe_allow_html=True)

    # Upload zone
    st.markdown("""
    <div class="upload-zone">
        <h3>📁 Get Started</h3>
        <p>Upload a PDF below to start chatting with your document</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("🔄 Processing your document... Splitting, embedding, and indexing."):
                vectorstore = ingest_documents(file_path)
                st.session_state["vectorstore"] = vectorstore
                st.session_state["chain"] = create_rag_chain(vectorstore)

            st.success(f"✅ **{uploaded_file.name}** processed successfully!")
            st.button("💬 Start Chatting →", on_click=go_to_chat, type="primary", use_container_width=True)

        elif "chain" in st.session_state:
            st.info("📚 You have a previously loaded document ready.")
            st.button("💬 Continue Chatting →", on_click=go_to_chat, type="primary", use_container_width=True)

    # Divider
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Feature cards
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">📄</div>
            <h3>Upload PDFs</h3>
            <p>Drop any PDF document and it's instantly processed and indexed</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <h3>Semantic Search</h3>
            <p>Find answers based on meaning, not just keyword matching</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">💡</div>
            <h3>Smart Answers</h3>
            <p>AI reads the relevant sections and generates clear answers</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📎</div>
            <h3>Source Citations</h3>
            <p>Every answer includes page references so you can verify</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">💾</div>
            <h3>Persistent Memory</h3>
            <p>Documents stay indexed — no need to re-upload on restart</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🏠</div>
            <h3>Runs Locally</h3>
            <p>Your data never leaves your machine when using Ollama mode</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# CHAT PAGE
# ============================================================
elif st.session_state["page"] == "chat":

    # Top bar
    top_left, top_center, top_right = st.columns([1, 3, 1])
    with top_left:
        st.button("← Back", on_click=go_to_home)
    with top_center:
        st.markdown("<h2 style='text-align:center; margin:0; padding:0.5rem 0;'>💬 DocChat AI</h2>", unsafe_allow_html=True)
    with top_right:
        if st.button("🗑️ Clear"):
            st.session_state["messages"] = []
            st.rerun()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Show chat messages
    for message in st.session_state["messages"]:
        avatar = "🧑" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.popover("View Sources"):
                    for src in message["sources"]:
                        st.markdown(
                            f'<div class="source-card"><strong>Page {src["page"]}</strong> '
                            f'from <strong>{os.path.basename(src["source"])}</strong><br>'
                            f'<em>{src["content_preview"]}</em></div>',
                            unsafe_allow_html=True,
                        )

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        if "chain" not in st.session_state:
            with st.chat_message("assistant", avatar="🤖"):
                st.warning("⚠️ Please upload a PDF document first!")
            st.session_state["messages"].append({
                "role": "assistant",
                "content": "⚠️ Please upload a PDF document first!",
            })
        else:
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    result = ask_question(st.session_state["chain"], prompt)

                st.markdown(result["answer"])

                if result["sources"]:
                    with st.popover("View Sources"):
                        for src in result["sources"]:
                            st.markdown(
                                f'<div class="source-card"><strong>Page {src["page"]}</strong> '
                                f'from <strong>{os.path.basename(src["source"])}</strong><br>'
                                f'<em>{src["content_preview"]}</em></div>',
                                unsafe_allow_html=True,
                            )

            st.session_state["messages"].append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            })
