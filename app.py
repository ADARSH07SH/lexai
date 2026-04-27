"""
LexAI — Legal Document Intelligence Platform

Streamlit application providing four core capabilities:
  1. Conversational Q&A against uploaded legal documents
  2. One-click executive summary extraction
  3. Automated red-flag risk scanning
  4. Targeted clause extraction via keyword search
"""

import os
import warnings

# Suppress loud transformers/cv2 warnings on Windows
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

# Inject Streamlit Cloud secrets into OS env vars (must happen before any imports)
try:
    import streamlit as st
    for key in ["GROQ_API_KEY", "WEAVIATE_URL", "WEAVIATE_API_KEY"]:
        if key not in os.environ or not os.environ[key]:
            if key in st.secrets:
                os.environ[key] = st.secrets[key]
except Exception:
    pass

import re
from datetime import datetime
import streamlit as st

from ingestion.parser import read_pdf, chunk_document
from core.chains import LegalAIEngine
from core.config import AppConfig
from retrieval.weaviate_client import WeaviateRetriever
from retrieval.memory_store import InMemoryRetriever


@st.cache_resource
def get_ai_engine():
    """Cache the LLM Engine so it isn't recreated on every render."""
    return LegalAIEngine()


@st.cache_resource
def get_retriever():
    """Cache the Vector Retriever connection."""
    try:
        retriever = WeaviateRetriever()
        return retriever, "Weaviate Cloud"
    except Exception as e:
        print(f"[LexAI] Weaviate connection failed: {e}")
        return InMemoryRetriever(), "In-Memory Fallback"


def format_collection_name(filename: str) -> str:
    """Format a valid Weaviate class name from the PDF filename and current time."""
    clean_name = re.sub(r'[^a-zA-Z0-9_]', '', filename.replace(".pdf", "").replace(".PDF", ""))
    if not clean_name or not clean_name[0].isalpha():
        clean_name = "Doc" + clean_name
    clean_name = clean_name[0].upper() + clean_name[1:]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"Col_{clean_name}_{timestamp}"


def run():
    """Main entry-point for the Streamlit UI."""

    st.set_page_config(
        page_title="LexAI — Legal Document Intelligence",
        layout="wide",
        page_icon="⚖️",
    )

    st.markdown(
        "<h1 style='text-align:center; font-weight:700; "
        "font-family:sans-serif; padding-top:0;'>"
        "⚖️ LexAI &mdash; Legal Document Intelligence</h1>",
        unsafe_allow_html=True,
    )

    # ── Session bootstrap ────────────────────────────────────────────
    if "doc_ready" not in st.session_state:
        st.session_state["doc_ready"] = False
        st.session_state["chat_history"] = []
        st.session_state["full_text"] = ""
        st.session_state["pdf_name"] = ""
        st.session_state["collection_id"] = ""
        st.session_state["suggestions"] = {
            "questions": ["Who are the parties involved?", "What is the governing law?", "What are the termination conditions?"],
            "clauses": ["Termination", "Liability", "Confidentiality"]
        }

    # ── Sidebar: PDF upload ──────────────────────────────────────────
    pdf_file = st.sidebar.file_uploader(
        label="Upload a legal document (PDF)", type="pdf"
    )

    ai = get_ai_engine()
    retriever, backend_label = get_retriever()

    if pdf_file:
        # Detect new file upload to reset state
        if st.session_state["pdf_name"] != pdf_file.name:
            st.session_state["pdf_name"] = pdf_file.name
            st.session_state["collection_id"] = format_collection_name(pdf_file.name)
            st.session_state["doc_ready"] = False
            st.session_state["chat_history"] = []
            st.session_state["full_text"] = ""

        col_name = st.session_state["collection_id"]

        # Process the uploaded PDF once per upload
        if not st.session_state["doc_ready"]:
            with st.spinner(f"Parsing document & building knowledge base…"):
                raw_text = read_pdf(pdf_file)
                st.session_state["full_text"] = raw_text
                sections = chunk_document(raw_text)

                retriever.create_collection(name=col_name)
                retriever.index_documents(
                    collection_name=col_name, sections=sections
                )
                
            # Generate AI Suggestions based on the text
            with st.spinner("Generating document insights…"):
                st.session_state["suggestions"] = ai.generate_suggestions(
                    raw_text, ai.build_suggestion_chain()
                )
                st.session_state["doc_ready"] = True

        # ── Tabs ─────────────────────────────────────────────────────
        tab_chat, tab_summary, tab_risk, tab_clause = st.tabs(
            ["💬 Chat Assistant", "⚡ Executive Summary",
             "🚩 Risk Scanner", "🔍 Clause Extractor"]
        )

        # ── Tab 1 — Conversational Q&A ───────────────────────────────
        with tab_chat:
            st.subheader("Chat with the Document")
            st.markdown("Ask anything about the uploaded legal document.")

            # Dynamic Suggestion Chips
            st.markdown("**Suggested Questions:**")
            sug_col1, sug_col2, sug_col3 = st.columns(3)
            active_query = None
            
            sugs = st.session_state["suggestions"]["questions"]
            
            if len(sugs) > 0 and sug_col1.button(sugs[0]):
                active_query = sugs[0]
            if len(sugs) > 1 and sug_col2.button(sugs[1]):
                active_query = sugs[1]
            if len(sugs) > 2 and sug_col3.button(sugs[2]):
                active_query = sugs[2]

            # Auto-scrolling chat container
            chat_container = st.container(height=400)
            
            with chat_container:
                for msg in st.session_state["chat_history"]:
                    st.chat_message(msg["role"]).write(msg["content"])

            # Standard chat input
            user_input = st.chat_input("e.g. What are the payment terms?")
            if user_input:
                active_query = user_input

            if active_query:
                st.session_state["chat_history"].append({"role": "user", "content": active_query})
                with chat_container:
                    st.chat_message("user").write(active_query)

                with st.spinner("Retrieving document chunks and analyzing…"):
                    # Extract keywords from the natural language question for better BM25 search
                    search_query = ai.extract_search_keywords(active_query, ai.build_keyword_extraction_chain())
                    
                    hits = retriever.keyword_search(query=search_query, collection_name=col_name)
                    
                    if not hits:
                        reply = "I cannot find any relevant sections in the document for this query."
                    else:
                        reply, _ = ai.answer_question(hits, active_query, ai.build_qa_chain())
                    
                    st.session_state["chat_history"].append({
                        "role": "assistant", 
                        "content": reply
                    })
                    
                st.rerun()

        # ── Tab 2 — Executive Summary ────────────────────────────────
        with tab_summary:
            st.subheader("Executive Summary")
            st.markdown(
                "Instantly extract parties, dates, financials, "
                "and a general overview."
            )
            if st.button("Generate Executive Summary"):
                with st.spinner("Extracting key details…"):
                    summary, _ = ai.generate_summary(
                        st.session_state["full_text"],
                        ai.build_summary_chain(),
                    )
                    st.markdown(summary)

        # ── Tab 3 — Risk Scanner ─────────────────────────────────────
        with tab_risk:
            st.subheader("Red-Flag Risk Analysis")
            st.markdown(
                "Uncover hidden auto-renewals, unlimited liabilities, "
                "or asymmetrical rights."
            )
            if st.button("Run Risk Scanner"):
                with st.spinner("Scanning for risks…"):
                    risks, _ = ai.scan_risks(
                        st.session_state["full_text"],
                        ai.build_risk_chain(),
                    )
                    st.markdown(risks)

        # ── Tab 4 — Clause Extractor ─────────────────────────────────
        with tab_clause:
            st.subheader("Targeted Clause Extractor")
            
            # Dynamic Clause Suggestions
            c_sugs = ", ".join([f"`{c}`" for c in st.session_state["suggestions"]["clauses"]])
            st.markdown(f"**Suggested Clauses for this document:** {c_sugs}")

            num_clauses = st.slider(
                "Number of clauses to extract", min_value=1, max_value=5
            )
            clause_map = {}
            columns = st.columns(num_clauses)

            for idx, col in enumerate(columns):
                # Auto-fill with suggestions if available
                default_kw = ""
                if idx < len(st.session_state["suggestions"]["clauses"]):
                    default_kw = st.session_state["suggestions"]["clauses"][idx]
                    
                keyword = col.text_input(
                    f"Clause {idx + 1} keyword",
                    value=default_kw,
                    key=f"kw_{idx}",
                )
                clause_map[keyword] = []

            if st.button("Extract Clauses"):
                with st.spinner("Extracting clauses…"):
                    chain = ai.build_clause_chain()
                    for keyword, results in clause_map.items():
                        if not keyword.strip():
                            continue
                        
                        hits = retriever.keyword_search(
                            query=keyword, collection_name=col_name
                        )
                        parsed, _ = ai.extract_clause(
                            relevant_docs=hits,
                            query=keyword,
                            chain=chain,
                        )
                        results.append(parsed)

                    entries = list(clause_map.items())
                    for idx, col in enumerate(columns):
                        kw, vals = entries[idx]
                        if not kw.strip():
                            continue
                        col.subheader(kw)
                        
                        text = "".join(vals)
                        col.text_area(
                            "Summary",
                            text,
                            key=f"out_{idx}",
                            height=300,
                        )

    else:
        st.info("👈 Upload a PDF from the sidebar to get started.")


if __name__ == "__main__":
    run()
