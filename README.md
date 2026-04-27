# LexAI — Legal Document Intelligence

AI-powered legal document analysis platform built with RAG (Retrieval-Augmented Generation) and LLMs.

Upload any legal PDF — contracts, court judgments, NDAs — and interact with it using natural language.

## Features

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
├── app.py                  # Streamlit entry point
├── core/
│   ├── config.py           # Environment & app configuration
│   └── chains.py           # LLM prompt chains (summary, QA, risk, clause)
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
