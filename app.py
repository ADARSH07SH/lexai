"""
LexAI — Legal Document Intelligence Platform

Streamlit application providing four core capabilities:
  1. Conversational Q&A against uploaded legal documents
  2. One-click executive summary extraction
  3. Automated red-flag risk scanning
  4. Targeted clause extraction via keyword search
"""

import uuid
import streamlit as st

from ingestion.parser import read_pdf, chunk_document
from core.chains import LegalAIEngine
from core.config import AppConfig
from retrieval.weaviate_client import WeaviateRetriever
from retrieval.memory_store import InMemoryRetriever


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
    if "collection_id" not in st.session_state:
        st.session_state["collection_id"] = "Col_" + str(uuid.uuid4()).replace("-", "")
        st.session_state["doc_ready"] = False
        st.session_state["chat_history"] = []
        st.session_state["full_text"] = ""

    # ── Sidebar: PDF upload ──────────────────────────────────────────
    pdf_file = st.sidebar.file_uploader(
        label="Upload a legal document (PDF)", type="pdf"
    )

    ai = LegalAIEngine()

    if pdf_file:
        # Initialise retriever (Weaviate → fallback to in-memory)
        try:
            retriever = WeaviateRetriever()
            backend_label = "Weaviate Cloud"
        except Exception:
            st.warning(
                "⚠️ Weaviate connection unavailable — using in-memory store "
                "(limited functionality)."
            )
            retriever = InMemoryRetriever()
            backend_label = "In-Memory"

        col_name = st.session_state["collection_id"]

        # Process the uploaded PDF once per session
        if not st.session_state["doc_ready"]:
            with st.spinner(
                f"Parsing document & building knowledge base ({backend_label})…"
            ):
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

            for msg in st.session_state["chat_history"]:
                st.chat_message(msg["role"]).write(msg["content"])

            user_input = st.chat_input(
                "e.g. What are the termination conditions?"
            )
            if user_input:
                st.session_state["chat_history"].append(
                    {"role": "user", "content": user_input}
                )
                st.chat_message("user").write(user_input)

                with st.spinner("Reviewing document…"):
                    ctx = [
                        st.session_state["full_text"][
                            : AppConfig.CONTEXT_CHAR_LIMIT
                        ]
                    ]
                    reply = ai.answer_question(
                        ctx, user_input, ai.build_qa_chain()
                    )
                    st.session_state["chat_history"].append(
                        {"role": "assistant", "content": reply}
                    )
                    st.chat_message("assistant").write(reply)

        # ── Tab 2 — Executive Summary ────────────────────────────────
        with tab_summary:
            st.subheader("Executive Summary")
            st.markdown(
                "Instantly extract parties, dates, financials, "
                "and a general overview."
            )
            if st.button("Generate Executive Summary"):
                with st.spinner("Extracting key details…"):
                    summary = ai.generate_summary(
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
                    risks = ai.scan_risks(
                        st.session_state["full_text"],
                        ai.build_risk_chain(),
                    )
                    st.markdown(risks)

        # ── Tab 4 — Clause Extractor ─────────────────────────────────
        with tab_clause:
            st.subheader("Targeted Clause Extractor")

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
                        parsed = ai.extract_clause(
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
                        col.text_area(
                            "Summary",
                            "".join(vals),
                            key=f"out_{idx}",
                            height=300,
                        )

    else:
        st.info("👈 Upload a PDF from the sidebar to get started.")


if __name__ == "__main__":
    run()
