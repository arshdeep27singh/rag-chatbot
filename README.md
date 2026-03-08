# 📄 RAG Chatbot — Chat with Your Documents

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload PDFs and ask questions about their content. Built with LangChain, ChromaDB, and OpenAI.

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│   PDF/Docs   │────▶│  Document Loader  │────▶│  Text Chunks │
└──────────────┘     │  & Splitter       │     └──────┬───────┘
                     └──────────────────┘            │
                                                     ▼
                                            ┌──────────────────┐
                                            │  Embedding Model  │
                                            │  (OpenAI / Local) │
                                            └──────┬───────────┘
                                                   │
                                                   ▼
┌──────────────┐     ┌──────────────────┐   ┌──────────────────┐
│   User       │────▶│  Streamlit UI    │   │  ChromaDB Vector │
│   Question   │     │                  │   │  Store           │
└──────────────┘     └──────┬───────────┘   └──────┬───────────┘
                            │                      │
                            ▼                      │
                     ┌──────────────────┐          │
                     │  Retrieval Chain  │◀─────────┘
                     │  (LangChain)     │
                     └──────┬───────────┘
                            │
                            ▼
                     ┌──────────────────┐
                     │  LLM (GPT-4 /   │
                     │  Ollama local)   │
                     └──────┬───────────┘
                            │
                            ▼
                     ┌──────────────────┐
                     │  Answer with     │
                     │  Source Citations │
                     └──────────────────┘
```

## Features

- 📄 Upload PDF documents and chat with their content
- 🔍 Semantic search using vector embeddings
- 📎 Source citations — see exactly where answers come from
- 💾 Persistent vector store — no re-processing on restart
- 🏠 Works with OpenAI API or local models via Ollama

## Tech Stack

| Component        | Technology                  |
|------------------|-----------------------------|
| Framework        | LangChain                   |
| Vector Store     | ChromaDB                    |
| Embeddings       | OpenAI / HuggingFace        |
| LLM              | GPT-4o / Ollama (Llama 3)   |
| UI               | Streamlit                   |
| Document Parsing | PyPDFLoader                 |

## Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key (or Ollama installed for local mode)

### Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Run the App

```bash
streamlit run src/app.py
```

Then open http://localhost:8501 in your browser.

### Using Local Models (Free, No API Key)

1. Install [Ollama](https://ollama.ai)
2. Pull a model: `ollama pull llama3`
3. Set `USE_LOCAL_MODEL=true` in your `.env` file
4. Run the app — it will use Ollama instead of OpenAI

## Project Structure

```
rag-chatbot/
├── src/
│   ├── app.py              # Streamlit UI
│   ├── ingest.py           # Document loading & chunking
│   ├── chain.py            # RAG chain setup
│   └── config.py           # Configuration
├── data/                   # Uploaded documents (gitignored)
├── tests/
│   └── test_chain.py       # Unit tests
├── .env.example            # Environment template
├── .gitignore
├── requirements.txt
└── README.md
```

## How It Works

1. **Ingest**: PDFs are loaded, split into chunks (~500 tokens each), and embedded into vectors
2. **Store**: Vectors are stored in ChromaDB for fast similarity search
3. **Retrieve**: When you ask a question, the most relevant chunks are retrieved
4. **Generate**: The LLM generates an answer using the retrieved context, with source citations

## License

MIT
