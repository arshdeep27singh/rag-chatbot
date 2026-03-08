from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma

from config import USE_LOCAL_MODEL, OPENAI_API_KEY, OLLAMA_MODEL


SYSTEM_TEMPLATE = """You are a helpful assistant that answers questions based on the provided documents.
Use the following context to answer the question. If the answer is not in the context,
say "I don't have enough information in the uploaded documents to answer this question."

Always cite which part of the document your answer comes from.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=SYSTEM_TEMPLATE,
)


def get_llm():
    """Return LLM based on configuration."""
    if USE_LOCAL_MODEL:
        return Ollama(model=OLLAMA_MODEL)
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        api_key=OPENAI_API_KEY,
    )


def create_rag_chain(vectorstore: Chroma) -> ConversationalRetrievalChain:
    """Create a conversational RAG chain."""
    llm = get_llm()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )
    return chain


def ask_question(chain: ConversationalRetrievalChain, question: str) -> dict:
    """Ask a question and return the answer with sources."""
    result = chain.invoke({"question": question})

    sources = []
    for doc in result.get("source_documents", []):
        source_info = {
            "page": doc.metadata.get("page", "N/A"),
            "source": doc.metadata.get("source", "Unknown"),
            "content_preview": doc.page_content[:200] + "...",
        }
        sources.append(source_info)

    return {
        "answer": result["answer"],
        "sources": sources,
    }
