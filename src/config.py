import os
from dotenv import load_dotenv

load_dotenv()


def _get_config(key: str, default: str = "") -> str:
    """Read from env vars first, then Streamlit secrets as fallback."""
    value = os.getenv(key, "")
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


GROQ_API_KEY = _get_config("GROQ_API_KEY")
LLM_PROVIDER = _get_config("LLM_PROVIDER", "groq").lower()  # "groq" or "ollama"
OLLAMA_MODEL = _get_config("OLLAMA_MODEL", "llama3")
GROQ_MODEL = _get_config("GROQ_MODEL", "llama-3.3-70b-versatile")
CHUNK_SIZE = int(_get_config("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(_get_config("CHUNK_OVERLAP", "50"))

CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "uploads")
