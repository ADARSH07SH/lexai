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
import streamlit.components.v1 as components


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
                print("[LexAI] Starting suggestion generation...")
                st.session_state["suggestions"] = ai.generate_suggestions(
                    raw_text, ai.build_suggestion_chain()
                )
                print(f"[LexAI] Suggestions ready: {st.session_state['suggestions']}")
                st.session_state["doc_ready"] = True

        # ── Tabs ─────────────────────────────────────────────────────
        tab_chat, tab_summary, tab_risk, tab_adversarial, tab_clause, tab_timeline = st.tabs(
            ["🤖 Chat Assistant", "📋 Executive Summary",
             "🛡️ Risk Scanner", "⚔️ Adversarial Detector", "📑 Clause Extractor", "📅 Timeline Extraction"]
        )

        # ── Tab 1 — Conversational Q&A ───────────────────────────────
        with tab_chat:
            st.subheader("Chat with the Document")
            st.markdown("Ask anything about the uploaded legal document.")

            # Auto-scrolling chat container
            chat_container = st.container(height=400)
            
            active_query = None
            with chat_container:
                for msg in st.session_state["chat_history"]:
                    st.chat_message(msg["role"]).write(msg["content"])
                    
                # Show dynamic suggestions INSIDE the chat container, only if empty
                if len(st.session_state["chat_history"]) == 0:
                    st.chat_message("assistant").write("Hello! I've analyzed the document. Here are some questions you can ask me:")
                    sugs = st.session_state["suggestions"]["questions"]
                    
                    sug_col1, sug_col2, sug_col3 = st.columns(3)
                    if len(sugs) > 0 and sug_col1.button(sugs[0]):
                        active_query = sugs[0]
                    if len(sugs) > 1 and sug_col2.button(sugs[1]):
                        active_query = sugs[1]
                    if len(sugs) > 2 and sug_col3.button(sugs[2]):
                        active_query = sugs[2]

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

        # ── Tab 3.5 — Adversarial Detector ─────────────────────────────
        with tab_adversarial:
            st.subheader("Adversarial Clause Detector")
            st.markdown(
                "Play **Devil's Advocate**. Discover how the opposing party could "
                "exploit these clauses against you, and how to defend yourself."
            )
            if st.button("Run Adversarial Detector"):
                with st.spinner("Tearing the contract apart…"):
                    adv_analysis, _ = ai.run_adversarial(
                        st.session_state["full_text"],
                        ai.build_adversarial_chain(),
                    )
                    st.markdown(adv_analysis)

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

        # ── Tab 5 — Timeline Extraction ────────────────────────────────
        with tab_timeline:
            if st.button("📅 Extract Timeline"):
                with st.spinner("Scanning document for dates and events…"):
                    timeline_raw, _ = ai.extract_timeline(
                        st.session_state["full_text"],
                        ai.build_timeline_chain(),
                    )
                    
                    # Parse the LLM output into timeline entries
                    entries = []
                    summary_text = ""
                    in_summary = False
                    for line in timeline_raw.split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                        if "SUMMARY:" in line:
                            in_summary = True
                            continue
                        if "TIMELINE:" in line:
                            continue
                        if in_summary:
                            summary_text += line + " "
                            continue
                        match = re.match(r'^(?:\d+\.)?\s*(.+?)\s*[—–-]\s*(.+)$', line)
                        if match:
                            entries.append({"date": match.group(1).strip(), "event": match.group(2).strip()})
                    
                    # Color palette for timeline nodes
                    colors = ["#4F8EF7", "#34D399", "#F59E0B", "#EF4444", "#A78BFA", "#EC4899", "#06B6D4", "#F97316"]
                    
                    if entries:
                        # Build visual HTML timeline
                        html = """
                        <style>
                        .tl-container { position: relative; padding: 20px 0; margin-left: 30px; }
                        .tl-line { position: absolute; left: 18px; top: 0; bottom: 0; width: 3px; background: linear-gradient(180deg, #4F8EF7 0%, #A78BFA 50%, #EC4899 100%); border-radius: 2px; }
                        .tl-item { position: relative; margin-bottom: 28px; padding-left: 50px; transition: transform 0.2s ease; }
                        .tl-item:hover { transform: translateX(6px); }
                        .tl-dot { position: absolute; left: 8px; top: 6px; width: 22px; height: 22px; border-radius: 50%; border: 3px solid #fff; box-shadow: 0 0 0 2px rgba(0,0,0,0.1), 0 2px 8px rgba(0,0,0,0.15); transition: transform 0.2s ease, box-shadow 0.2s ease; }
                        .tl-item:hover .tl-dot { transform: scale(1.3); box-shadow: 0 0 0 3px rgba(79,142,247,0.3), 0 4px 12px rgba(0,0,0,0.2); }
                        .tl-date { font-weight: 700; font-size: 15px; color: #E2E8F0; margin-bottom: 2px; font-family: 'Segoe UI', sans-serif; }
                        .tl-event { font-size: 14px; color: #94A3B8; line-height: 1.5; font-family: 'Segoe UI', sans-serif; }
                        .tl-card { background: rgba(30, 41, 59, 0.7); border-radius: 10px; padding: 12px 16px; border-left: 3px solid; transition: background 0.2s ease, border-color 0.2s ease; }
                        .tl-item:hover .tl-card { background: rgba(51, 65, 85, 0.8); }
                        .tl-summary { background: rgba(30, 41, 59, 0.5); border-radius: 10px; padding: 16px; margin-top: 20px; margin-left: 30px; border: 1px solid rgba(148, 163, 184, 0.2); }
                        .tl-summary p { color: #94A3B8; font-size: 14px; line-height: 1.6; margin: 0; font-family: 'Segoe UI', sans-serif; }
                        .tl-summary-title { color: #E2E8F0; font-weight: 700; font-size: 14px; margin-bottom: 8px; font-family: 'Segoe UI', sans-serif; }
                        </style>
                        <div class="tl-container"><div class="tl-line"></div>
                        """
                        for i, entry in enumerate(entries):
                            color = colors[i % len(colors)]
                            warning = "⚠️ " if "⚠️" in entry["date"] or "⚠️" in entry["event"] else ""
                            date_clean = entry["date"].replace("⚠️", "").strip()
                            event_clean = entry["event"].replace("⚠️", "").strip()
                            html += f"""
                            <div class="tl-item">
                                <div class="tl-dot" style="background: {color};"></div>
                                <div class="tl-card" style="border-left-color: {color};">
                                    <div class="tl-date">{warning}{date_clean}</div>
                                    <div class="tl-event">{event_clean}</div>
                                </div>
                            </div>
                            """
                        html += "</div>"
                        
                        if summary_text.strip():
                            html += f"""
                            <div class="tl-summary">
                                <div class="tl-summary-title">📝 Timeline Summary</div>
                                <p>{summary_text.strip()}</p>
                            </div>
                            """
                        
                        # Calculate height based on number of entries
                        height = len(entries) * 80 + 200
                        components.html(html, height=height, scrolling=False)
                    else:
                        st.markdown(timeline_raw)

    else:
        st.info("👈 Upload a PDF from the sidebar to get started.")


if __name__ == "__main__":
    run()
