import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config import CHUNK_SIZE, CHUNK_OVERLAP


def get_embeddings():
    """Return HuggingFace embedding model (free, no API key needed)."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


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
    """Ingest a PDF into an in-memory vector store and return it."""
    chunks = load_and_split_pdf(file_path)
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )
    return vectorstore
