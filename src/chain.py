from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma

from config import LLM_PROVIDER, GROQ_API_KEY, OLLAMA_MODEL, GROQ_MODEL


SYSTEM_TEMPLATE = """Use the following pieces of context to answer the question. Base your answer only on the context provided.

Context:
{context}

Question: {question}

Helpful Answer:"""

PROMPT = ChatPromptTemplate.from_template(SYSTEM_TEMPLATE)


def get_llm():
    """Return LLM based on configuration."""
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import OllamaLLM
        return OllamaLLM(model=OLLAMA_MODEL)
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0.2,
        api_key=GROQ_API_KEY,
    )


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(vectorstore: Chroma):
    """Create a RAG chain using LCEL."""
    llm = get_llm()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return {"chain": chain, "retriever": retriever}


def ask_question(rag, question: str) -> dict:
    """Ask a question and return the answer with sources."""
    source_docs = rag["retriever"].invoke(question)
    answer = rag["chain"].invoke(question)

    sources = []
    for doc in source_docs:
        source_info = {
            "page": doc.metadata.get("page", "N/A"),
            "source": doc.metadata.get("source", "Unknown"),
            "content_preview": doc.page_content[:200] + "...",
        }
        sources.append(source_info)

    return {
        "answer": answer,
        "sources": sources,
    }
