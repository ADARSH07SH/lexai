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
        self._llm = ChatGroq(
            model_name=AppConfig.DEFAULT_MODEL,
            api_key=self._cfg.GROQ_API_KEY
        )

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

    def build_suggestion_chain(self):
        """Build a chain to generate dynamic questions and clauses."""
        template = """
        You are an expert legal assistant. Read the provided Document Text.
        Generate exactly 3 insightful questions the user could ask about this document,
        and exactly 3 key legal clauses (1-3 words each, e.g. "Termination", "Confidentiality")
        that would be useful to extract from this document.
        
        Output STRICTLY in the following format with NO other text:
        QUESTIONS:
        1. [Question 1]
        2. [Question 2]
        3. [Question 3]
        CLAUSES:
        1. [Clause 1]
        2. [Clause 2]
        3. [Clause 3]

        Document Text:
        {context}
        """
        return (
            PromptTemplate(template=template, input_variables=["context"])
            | self._llm
        )

    def build_keyword_extraction_chain(self):
        """Build a chain to extract BM25 search keywords from a chat query."""
        template = """
        You are a search query generator. Given the user's question, extract the 3 to 5 most important 
        keywords or short phrases to search for in a legal document database.
        Output ONLY the keywords separated by spaces. Do NOT output any conversational text.
        
        Question: {question}
        Keywords:"""
        return PromptTemplate(template=template, input_variables=["question"]) | self._llm

    # ------------------------------------------------------------------
    # High-level invocation helpers
    # ------------------------------------------------------------------

    def extract_search_keywords(self, question: str, chain) -> str:
        """Convert a natural language question into BM25 keywords."""
        if len(question.split()) < 4:
            return question
        return chain.invoke({"question": question}).content.strip()

    def generate_suggestions(self, document_text: str, chain) -> dict:
        """Generate and parse dynamic suggestions."""
        inputs = {"context": document_text[: AppConfig.CONTEXT_CHAR_LIMIT]}
        raw = chain.invoke(inputs).content
        
        # Parse the raw text into a dictionary
        res = {"questions": [], "clauses": []}
        mode = None
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            if "QUESTIONS:" in line:
                mode = "q"
                continue
            if "CLAUSES:" in line:
                mode = "c"
                continue
            
            # Match "1. Text" or "- Text"
            match = re.match(r"^(?:\d+\.|-)\s*(.+)$", line)
            if match:
                val = match.group(1).replace("[", "").replace("]", "").replace('"', '').strip()
                if mode == "q" and len(res["questions"]) < 3:
                    res["questions"].append(val)
                elif mode == "c" and len(res["clauses"]) < 3:
                    res["clauses"].append(val)
                    
        # Fallbacks if LLM fails formatting
        if not res["questions"]:
            res["questions"] = ["Who are the parties involved?", "What is the governing law?", "What are the termination conditions?"]
        if not res["clauses"]:
            res["clauses"] = ["Termination", "Liability", "Confidentiality"]
            
        return res

    def extract_clause(self, relevant_docs: list, query: str, chain) -> tuple[str, str]:
        """Run the clause-extraction chain, clean the output, and log."""
        context_str = (
            "\n".join(relevant_docs)
            if isinstance(relevant_docs, list)
            else str(relevant_docs)
        )

        inputs = {"context": context_str, "clause": query}
        
        # Log to console
        prompt_val = chain.first.invoke(inputs).to_string()
        print("\n" + "="*50)
        print("🧠 LLM PROMPT (Clause Extractor):")
        print(prompt_val)
        print("="*50 + "\n")

        raw = chain.invoke(inputs)
        text = raw.content

        # Clean output
        text = re.sub(r"<.*?>", "", text).replace("\n", " ")
        if re.search(r"Clause Not Found", text):
            return "Clause Not Found", prompt_val

        for pattern in [r"^(.*?)```", r'^(.*?)"""', r"^(.*?)'''"]:
            match = re.search(pattern, text)
            if match:
                text = match.group(1)

        return text.strip(), prompt_val

    def generate_summary(self, document_text: str, chain) -> tuple[str, str]:
        """Generate an executive summary from the first N characters."""
        inputs = {"context": document_text[: AppConfig.CONTEXT_CHAR_LIMIT]}
        
        prompt_val = chain.first.invoke(inputs).to_string()
        print("\n" + "="*50)
        print("🧠 LLM PROMPT (Executive Summary):")
        print(prompt_val)
        print("="*50 + "\n")
        
        result = chain.invoke(inputs)
        return result.content, prompt_val

    def answer_question(self, doc_chunks: list, question: str, chain) -> tuple[str, str]:
        """Answer a user's question using retrieved document chunks."""
        context_str = "\n".join(doc_chunks)
        inputs = {"context": context_str, "question": question}
        
        prompt_val = chain.first.invoke(inputs).to_string()
        print("\n" + "="*50)
        print("🧠 LLM PROMPT (Chat Assistant):")
        print(prompt_val)
        print("="*50 + "\n")
        
        result = chain.invoke(inputs)
        return result.content, prompt_val

    def scan_risks(self, document_text: str, chain) -> tuple[str, str]:
        """Scan the document for contractual red flags."""
        inputs = {"context": document_text[: AppConfig.CONTEXT_CHAR_LIMIT]}
        
        prompt_val = chain.first.invoke(inputs).to_string()
        print("\n" + "="*50)
        print("🧠 LLM PROMPT (Risk Scanner):")
        print(prompt_val)
        print("="*50 + "\n")
        
        result = chain.invoke(inputs)
        return result.content, prompt_val
