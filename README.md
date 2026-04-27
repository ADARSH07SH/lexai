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

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| LLM Inference | Groq Cloud (Llama 3.1 8B) |
| Vector Search | Weaviate (BM25) |
| Document Parsing | PyPDF2 + pytesseract (OCR) |
| Orchestration | LangChain |

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
