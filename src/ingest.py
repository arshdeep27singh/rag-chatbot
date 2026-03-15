import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config import CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_PERSIST_DIR, LLM_PROVIDER, OPENAI_API_KEY


def get_embeddings():
    """Return embedding model based on configuration."""
    if LLM_PROVIDER == "ollama":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return OpenAIEmbeddings(api_key=OPENAI_API_KEY)


def load_and_split_pdf(file_path: str) -> list:
    """Load a PDF and split it into chunks."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    return chunks


def ingest_documents(file_path: str) -> Chroma:
    """Ingest a PDF into the vector store and return the store."""
    chunks = load_and_split_pdf(file_path)
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return vectorstore


def get_vectorstore() -> Chroma | None:
    """Load existing vector store if it exists."""
    if not os.path.exists(CHROMA_PERSIST_DIR):
        return None

    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )
