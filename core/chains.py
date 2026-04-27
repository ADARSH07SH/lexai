"""
LegalAIEngine — Central AI/LLM orchestration layer.

Houses all LangChain prompt templates and chain-building logic for:
  - Clause extraction via keyword-targeted RAG
  - Executive document summarization
  - Conversational Q&A against uploaded documents
  - Red-flag risk scanning
"""

import re
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from core.config import AppConfig


class LegalAIEngine:
    """Manages all LLM interactions for legal document analysis."""

    def __init__(self) -> None:
        self._cfg = AppConfig()
        self._llm = ChatGroq(model_name=AppConfig.DEFAULT_MODEL)

    # ------------------------------------------------------------------
    # Chain builders
    # ------------------------------------------------------------------

    def build_clause_chain(self):
        """Build a chain that summarises a specific lease/contract clause."""
        template = """
        You are a lease clause summarization expert. Summarize the lease
        clauses from the Context below in a single, concise paragraph.
        The summary must satisfy the definition and guidelines for the
        requested clause.

        If the context mentions a section number, page number, or clause
        reference, include that information in parentheses at the end.

        If the summary is irrelevant to `{clause}`, respond only with
        'Clause Not Found'. Do not fabricate information.

        Context:
        {context}

        Clause: {clause}

        Answer:"""

        prompt = PromptTemplate(
            template=template, input_variables=["context", "clause"]
        )
        return prompt | self._llm

    def build_summary_chain(self):
        """Build a chain that produces an executive summary of the document."""
        template = """
        You are an elite legal analyst. Extract the key executive details
        from the provided legal document text. First determine the overall
        document TYPE (e.g., Court Case Judgement, Commercial Lease, NDA,
        Employment Contract). Then extract the following using Markdown:

        * **Document Type**: (Judicial Decision, Lease Agreement, etc.)
        * **Primary Parties/Entities**: Main entities involved
        * **Key Dates & Financials**: Important dates, numeric/financial values
        * **Summary Overview**: One-paragraph overview of the document

        If a detail is not applicable for this document type, state
        "Not applicable to this document type."

        Document Text:
        {context}
        """
        return (
            PromptTemplate(template=template, input_variables=["context"])
            | self._llm
        )

    def build_qa_chain(self):
        """Build a conversational Q&A chain grounded in document context."""
        template = """
        You are an expert AI legal assistant. Answer the user's question
        accurately based ONLY on the provided Context. If the Context does
        not contain the answer, politely say "I cannot find the answer to
        this in the document." Do not fabricate facts.

        Context:
        {context}

        Question: {question}

        Answer:"""
        return (
            PromptTemplate(
                template=template, input_variables=["context", "question"]
            )
            | self._llm
        )

    def build_risk_chain(self):
        """Build a chain that scans for red-flag contract risks."""
        template = """
        You are a highly experienced contract lawyer identifying Red Flags
        in legal documents. Review the clauses in the Context below.
        Identify high-risk terms such as:
        - Auto-renewal / evergreen clauses
        - Unlimited or severely asymmetrical liability
        - One-sided termination rights
        - Unreasonable financial penalties

        If red flags are found, list them and explain WHY they are risky
        in plain English. If none are found, state:
        "✅ No major red flags detected."

        Context:
        {context}

        Risk Analysis:"""
        return (
            PromptTemplate(template=template, input_variables=["context"])
            | self._llm
        )

    # ------------------------------------------------------------------
    # High-level invocation helpers
    # ------------------------------------------------------------------

    def extract_clause(self, relevant_docs: list, query: str, chain) -> str:
        """Run the clause-extraction chain and clean the LLM output."""
        context_str = (
            "\n".join(relevant_docs)
            if isinstance(relevant_docs, list)
            else str(relevant_docs)
        )

        raw = chain.invoke({"context": context_str, "clause": query})
        text = raw.content

        # Strip any residual HTML-like tags
        text = re.sub(r"<.*?>", "", text).replace("\n", " ")

        # If the model says "Clause Not Found", return that directly
        if re.search(r"Clause Not Found", text):
            return "Clause Not Found"

        # Trim anything after code-fence or triple-quote artifacts
        for pattern in [r"^(.*?)```", r'^(.*?)"""', r"^(.*?)'''"]:
            match = re.search(pattern, text)
            if match:
                text = match.group(1)

        return text.strip()

    def generate_summary(self, document_text: str, chain) -> str:
        """Generate an executive summary from the first N characters."""
        result = chain.invoke(
            {"context": document_text[: AppConfig.CONTEXT_CHAR_LIMIT]}
        )
        return result.content

    def answer_question(self, doc_chunks: list, question: str, chain) -> str:
        """Answer a user's question using retrieved document chunks."""
        context_str = "\n".join(doc_chunks)
        result = chain.invoke({"context": context_str, "question": question})
        return result.content

    def scan_risks(self, document_text: str, chain) -> str:
        """Scan the document for contractual red flags."""
        result = chain.invoke(
            {"context": document_text[: AppConfig.CONTEXT_CHAR_LIMIT]}
        )
        return result.content
