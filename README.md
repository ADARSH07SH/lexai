# LexAI v2 — Legal Document Intelligence

AI-powered legal document analysis platform built with true RAG (Retrieval-Augmented Generation) and LLMs.

Upload any legal PDF — contracts, court judgments, NDAs — and interact with it using natural language.

## ✨ New in v2
- **True RAG Chat**: The Chat Assistant now performs semantic searches across the entire document using Weaviate before answering, meaning it can answer questions no matter how long the document is.
- **Smart Suggestions**: One-click suggestion buttons for quick chats and common clause extractions (e.g., Termination, Liability).
- **LLM Prompt Transparency**: View the exact prompt and document context sent to the LLM directly in the UI or in the terminal logs.
- **Auto-Scrolling UI**: The Chat Assistant now features a fixed-height container that automatically scrolls to the newest message.
- **Dynamic Metadata**: The sidebar displays uploaded file sizes, upload timestamps, and active database engines. Weaviate collections are intelligently named based on the PDF filename.
- **Optimized Caching**: Heavy AI and database connections are cached using `@st.cache_resource` for instantaneous hot-reloads and better performance.

## Core Features

- **💬 Chat Assistant** — Conversational Q&A grounded in the uploaded document
- **⚡ Executive Summary** — One-click extraction of parties, dates, financials, and overview
- **🚩 Risk Scanner** — Automated detection of red-flag clauses (auto-renewals, liability traps, etc.)
- **🔍 Clause Extractor** — Targeted keyword-based clause retrieval and summarization

## How It Works (Architecture)

LexAI is built using a lightweight but powerful Legal AI pipeline:

- **1. Parsing & Chunking**: When you upload a PDF, PyPDF2 extracts the text. The document is then dynamically "chunked" based on headings, paragraphs, and sections to preserve legal context (done in `ingestion/parser.py`).
- **2. Embeddings / Vectorization**: To keep the system lightning-fast and offline-capable, LexAI **does not use heavy vector embeddings**. Instead, it uses **BM25 Keyword Scoring** natively via Weaviate Cloud (or a local dictionary fallback). This ensures highly precise keyword-matching for specific legal clauses.
- **3. Retrieval (RAG)**: When you ask a question, an LLM extracts the core keywords from your question, queries Weaviate via BM25, and retrieves the top 4 most relevant chunks from your document.
- **4. LLM Generation**: Those chunks are sent to **Groq Cloud (Llama 3.1 8B)** to generate a highly accurate, grounded answer.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend UI | Streamlit |
| LLM Engine | Groq Cloud (Llama 3.1 8B) |
| Database | Weaviate (BM25 search mode) |
| OCR / Extraction | PyPDF2 & Tesseract |
| Orchestrator | LangChain |

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/ADARSH07SH/lexai.git
cd lexai
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
copy .env.example .env
# Edit .env and add your Groq + Weaviate credentials
```

### 3. Run

```bash
streamlit run app.py
```

## Architecture

```
lexai/
├── app.py                  # Streamlit entry point (UI & caching)
├── core/
│   ├── config.py           # Environment & app configuration
│   └── chains.py           # LLM prompt chains and logging logic
├── ingestion/
│   └── parser.py           # PDF text extraction & document chunking
├── retrieval/
│   ├── weaviate_client.py  # Weaviate Cloud integration
│   └── memory_store.py     # In-memory fallback retriever
└── data/                   # Sample documents (gitignored)
```

## Requirements

- Python 3.10+
- Groq API Key ([console.groq.com](https://console.groq.com))
- Weaviate Cloud instance (optional — falls back to in-memory store)

## License

MIT
