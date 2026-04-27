"""
LexAI — Legal Document Intelligence Platform

Streamlit application providing four core capabilities:
  1. Conversational Q&A against uploaded legal documents
  2. One-click executive summary extraction
  3. Automated red-flag risk scanning
  4. Targeted clause extraction via keyword search
"""

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
        # Test connection by making a dummy collection
        return retriever, "Weaviate Cloud"
    except Exception as e:
        print(f"[LexAI] Weaviate connection failed: {e}")
        return InMemoryRetriever(), "In-Memory Fallback"


def format_collection_name(filename: str) -> str:
    """Format a valid Weaviate class name from the PDF filename and current time."""
    # Strip extension and invalid chars
    clean_name = re.sub(r'[^a-zA-Z0-9_]', '', filename.replace(".pdf", "").replace(".PDF", ""))
    
    # Weaviate class names must start with a capital letter
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
        st.session_state["upload_time"] = ""

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
            st.session_state["upload_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state["doc_ready"] = False
            st.session_state["chat_history"] = []
            st.session_state["full_text"] = ""

        col_name = st.session_state["collection_id"]

        # Sidebar Metadata Rendering
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📄 Document Info")
        st.sidebar.markdown(f"**Name:** `{pdf_file.name}`")
        st.sidebar.markdown(f"**Size:** `{pdf_file.size / 1024:.1f} KB`")
        st.sidebar.markdown(f"**Uploaded:** `{st.session_state['upload_time']}`")
        st.sidebar.markdown(f"**Engine:** `{backend_label}`")
        st.sidebar.markdown(f"**DB ID:** `{col_name}`")

        # Process the uploaded PDF once per upload
        if not st.session_state["doc_ready"]:
            with st.spinner(f"Parsing document & building knowledge base ({backend_label})…"):
                raw_text = read_pdf(pdf_file)
                st.session_state["full_text"] = raw_text
                sections = chunk_document(raw_text)

                retriever.create_collection(name=col_name)
                retriever.index_documents(
                    collection_name=col_name, sections=sections
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

            # Suggestion Chips
            st.markdown("**Suggested Questions:**")
            sug_col1, sug_col2, sug_col3 = st.columns(3)
            
            # Use a variable to track if a suggestion was clicked, or if chat_input was used
            active_query = None
            
            if sug_col1.button("Who are the parties involved?"):
                active_query = "Who are the primary parties involved in this document?"
            elif sug_col2.button("What is the governing law?"):
                active_query = "What is the governing law and jurisdiction?"
            elif sug_col3.button("What are the termination conditions?"):
                active_query = "What are the termination conditions and notice periods?"

            # Auto-scrolling chat container
            chat_container = st.container(height=400)
            
            with chat_container:
                for msg in st.session_state["chat_history"]:
                    st.chat_message(msg["role"]).write(msg["content"])
                    if "prompt" in msg:
                        with st.expander("🔍 View LLM Context & Prompt"):
                            st.code(msg["prompt"], language="text")

            # Standard chat input
            user_input = st.chat_input("e.g. What are the payment terms?")
            
            # Resolve which input to use
            if user_input:
                active_query = user_input

            if active_query:
                # Add user message to history and UI
                st.session_state["chat_history"].append({"role": "user", "content": active_query})
                with chat_container:
                    st.chat_message("user").write(active_query)

                # RAG Retrieval + LLM Answer
                with st.spinner("Retrieving document chunks and analyzing…"):
                    # Use Vector DB instead of dumping the whole text!
                    hits = retriever.keyword_search(query=active_query, collection_name=col_name)
                    
                    if not hits:
                        reply = "I cannot find any relevant sections in the document for this query."
                        prompt_val = "No context retrieved."
                    else:
                        reply, prompt_val = ai.answer_question(hits, active_query, ai.build_qa_chain())
                    
                    st.session_state["chat_history"].append({
                        "role": "assistant", 
                        "content": reply,
                        "prompt": prompt_val
                    })
                    
                # Rerun to update the chat_container with the new messages
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
                    summary, prompt_val = ai.generate_summary(
                        st.session_state["full_text"],
                        ai.build_summary_chain(),
                    )
                    st.markdown(summary)
                    with st.expander("🔍 View LLM Context & Prompt"):
                        st.code(prompt_val, language="text")

        # ── Tab 3 — Risk Scanner ─────────────────────────────────────
        with tab_risk:
            st.subheader("Red-Flag Risk Analysis")
            st.markdown(
                "Uncover hidden auto-renewals, unlimited liabilities, "
                "or asymmetrical rights."
            )
            if st.button("Run Risk Scanner"):
                with st.spinner("Scanning for risks…"):
                    risks, prompt_val = ai.scan_risks(
                        st.session_state["full_text"],
                        ai.build_risk_chain(),
                    )
                    st.markdown(risks)
                    with st.expander("🔍 View LLM Context & Prompt"):
                        st.code(prompt_val, language="text")

        # ── Tab 4 — Clause Extractor ─────────────────────────────────
        with tab_clause:
            st.subheader("Targeted Clause Extractor")
            
            st.markdown("**Common Clauses to Extract:** `Termination`, `Liability`, `Confidentiality`, `Indemnification`")

            num_clauses = st.slider(
                "Number of clauses to extract", min_value=1, max_value=5
            )
            clause_map = {}
            columns = st.columns(num_clauses)

            for idx, col in enumerate(columns):
                keyword = col.text_input(
                    f"Clause {idx + 1} keyword",
                    value="",
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
                        parsed, prompt_val = ai.extract_clause(
                            relevant_docs=hits,
                            query=keyword,
                            chain=chain,
                        )
                        results.append((parsed, prompt_val))

                    entries = list(clause_map.items())
                    for idx, col in enumerate(columns):
                        kw, vals = entries[idx]
                        if not kw.strip():
                            continue
                        col.subheader(kw)
                        
                        text = "".join([v[0] for v in vals])
                        prompts = "\n\n".join([v[1] for v in vals])
                        
                        col.text_area(
                            "Summary",
                            text,
                            key=f"out_{idx}",
                            height=300,
                        )
                        with col.expander("🔍 View Prompt"):
                            st.code(prompts, language="text")

    else:
        st.info("👈 Upload a PDF from the sidebar to get started.")


if __name__ == "__main__":
    run()
