# LexAI — Legal Document Intelligence Platform

AI-powered legal document analysis platform built with Retrieval-Augmented Generation (RAG) and Large Language Models.

Upload legal PDFs — contracts, court judgments, NDAs, lease agreements — and interact with them using natural language queries.

---

## Features

### 💬 Chat Assistant
Conversational Q&A interface with multiple persona modes (Corporate Lawyer, Judge, Layman, Adversarial Opponent). Ask questions about any aspect of your legal documents and get accurate, context-aware responses.

### 📋 Executive Summary
One-click extraction of key document details including parties involved, important dates, financial terms, and a comprehensive overview.

### 🛡️ Risk Scanner
Automated detection of red-flag clauses such as auto-renewal terms, unlimited liability provisions, one-sided termination rights, and unreasonable penalties.

### ⚔️ Adversarial Detector
Analyzes contracts from an opposing party's perspective to identify exploitable clauses and suggests defensive strategies.

### 📑 Clause Extractor
Targeted keyword-based clause retrieval with AI-powered summarization. Extract specific clauses like termination, confidentiality, or liability provisions.

### 📅 Timeline Extraction
Automatically identifies and visualizes all dates, deadlines, and time-bound events in chronological order with an interactive timeline interface.

### 📚 Multiple Document Support
Upload and analyze multiple legal documents simultaneously, creating a unified knowledge base for comprehensive analysis.

---

## Architecture

LexAI uses a modular architecture designed for legal document intelligence:

### System Components

```
┌─────────────────┐
│   Streamlit UI  │  ← User Interface Layer
└────────┬────────┘
         │
┌────────▼────────┐
│  LangChain      │  ← LLM Orchestration
│  Prompt Chains  │
└────────┬────────┘
         │
┌────────▼────────┐
│  Groq Cloud     │  ← LLM Inference (Llama 3.1)
│  (Llama 3.1)    │
└─────────────────┘

┌─────────────────┐
│  PDF Parser     │  ← Document Ingestion
│  (PyPDF2)       │
└────────┬────────┘
         │
┌────────▼────────┐
│  Text Chunker   │  ← Semantic Segmentation
└────────┬────────┘
         │
┌────────▼────────┐
│  Weaviate DB    │  ← Vector Storage & BM25 Search
│  (BM25 Search)  │
└─────────────────┘
```

### Processing Pipeline

1. **Document Ingestion** (`ingestion/parser.py`)
   - PDF text extraction using PyPDF2
   - OCR fallback for scanned documents (Tesseract)
   - Heading detection and section classification
   - Recursive text splitting with context preservation

2. **Knowledge Base Creation** (`retrieval/`)
   - Document chunking (1000 chars, 30 char overlap)
   - Indexing in Weaviate Cloud (or in-memory fallback)
   - BM25 keyword scoring for precise legal term matching

3. **Query Processing** (`core/chains.py`)
   - Keyword extraction from natural language queries
   - BM25 retrieval of relevant document chunks
   - Context assembly for LLM prompts

4. **AI Generation** (`core/chains.py`)
   - Groq Cloud API (Llama 3.1 8B Instant)
   - Specialized prompt templates for each feature
   - Grounded responses with source attribution

---

## Project Structure

```
lexai/
├── app.py                      # Main Streamlit application
├── core/
│   ├── config.py               # Configuration and environment variables
│   └── chains.py               # LLM prompt chains and AI engine
├── ingestion/
│   ├── parser.py               # PDF parsing and text chunking
│   └── __init__.py
├── retrieval/
│   ├── weaviate_client.py      # Weaviate Cloud integration
│   ├── memory_store.py         # In-memory fallback retriever
│   └── __init__.py
├── data/                       # Sample legal documents
│   ├── Sample_Lease.pdf
│   └── Sample_NDA.pdf
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
└── README.md                   # This file
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit |
| **LLM Provider** | Groq Cloud (Llama 3.1 8B Instant) |
| **Vector Database** | Weaviate Cloud |
| **Search Algorithm** | BM25 (keyword-based) |
| **PDF Parsing** | PyPDF2 |
| **OCR Engine** | Tesseract (optional) |
| **LLM Orchestration** | LangChain |
| **Language** | Python 3.10+ |

---

## Installation & Setup

### Prerequisites

- Python 3.10 or higher
- Groq API key ([Get one here](https://console.groq.com))
- Weaviate Cloud instance (optional, falls back to in-memory)

### Step 1: Clone the Repository

```bash
git clone https://github.com/ADARSH07SH/lexai.git
cd lexai
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
# Required:
GROQ_API_KEY=your_groq_api_key_here

# Optional (for Weaviate Cloud):
WEAVIATE_URL=your_weaviate_cluster_url
WEAVIATE_API_KEY=your_weaviate_api_key
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

---

## Usage Guide

### Uploading Documents

1. Click the file uploader in the sidebar
2. Select one or more PDF files (legal contracts, judgments, agreements)
3. Wait for the system to parse and index the documents
4. The system will generate smart suggestions based on document content

### Chat Assistant

1. Navigate to the "Chat Assistant" tab
2. Select a persona (Corporate Lawyer, Judge, Layman, Adversarial Opponent)
3. Type your question or click a suggested question
4. The AI will retrieve relevant sections and provide a grounded answer

### Executive Summary

1. Go to the "Executive Summary" tab
2. Click "Generate Executive Summary"
3. View extracted parties, dates, financials, and document overview

### Risk Scanner

1. Navigate to the "Risk Scanner" tab
2. Click "Run Risk Scanner"
3. Review identified red flags with explanations

### Adversarial Detector

1. Go to the "Adversarial Detector" tab
2. Click "Run Adversarial Detector"
3. See how opposing parties could exploit clauses and defensive strategies

### Clause Extractor

1. Navigate to the "Clause Extractor" tab
2. Use the slider to select number of clauses to extract
3. Enter keywords (or use suggested clauses)
4. Click "Extract Clauses" to get AI-powered summaries

### Timeline Extraction

1. Go to the "Timeline Extraction" tab
2. Click "Extract Timeline"
3. View an interactive chronological timeline of all dates and events

---

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Groq Cloud API key for LLM access | Yes |
| `WEAVIATE_URL` | Weaviate Cloud cluster URL | No |
| `WEAVIATE_API_KEY` | Weaviate Cloud API key | No |

### Model Configuration

Default model: `llama-3.1-8b-instant` (configurable in `core/config.py`)

### Context Limits

- Maximum context size: 30,000 characters
- Chunk size: 1,000 characters
- Chunk overlap: 30 characters

---

## Development

### Adding New Features

1. Create prompt templates in `core/chains.py`
2. Add UI components in `app.py`
3. Update session state management as needed

### Customizing Prompts

Edit the prompt templates in `core/chains.py` to adjust AI behavior for different legal domains or jurisdictions.

### Extending Retrieval

Modify `retrieval/weaviate_client.py` to add semantic search or hybrid retrieval strategies.

---

## Troubleshooting

### Weaviate Connection Failed

If Weaviate Cloud connection fails, the system automatically falls back to an in-memory retriever. Check your `WEAVIATE_URL` and `WEAVIATE_API_KEY` in `.env`.

### OCR Not Working

Install Tesseract OCR:
- Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- Linux: `sudo apt-get install tesseract-ocr`
- Mac: `brew install tesseract`

### API Rate Limits

Groq Cloud has rate limits. If you encounter errors, wait a few seconds and retry.

---

## Roadmap

- [ ] Multi-document comparison and cross-referencing
- [ ] Precedent and case law integration (Indian legal system)
- [ ] Compliance checking against statutory requirements
- [ ] Contract version diff and change tracking
- [ ] Multi-language support (Hindi, regional languages)
- [ ] Fine-tuned legal AI model for Indian law

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## License

MIT License - see LICENSE file for details

---

## Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) - Web framework
- [LangChain](https://langchain.com/) - LLM orchestration
- [Groq](https://groq.com/) - Fast LLM inference
- [Weaviate](https://weaviate.io/) - Vector database
- [PyPDF2](https://pypdf2.readthedocs.io/) - PDF parsing

---

## Contact

For questions or support, please open an issue on GitHub.
