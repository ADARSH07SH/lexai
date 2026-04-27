"""
In-memory document store — lightweight fallback retriever.

Used when Weaviate Cloud is unreachable. Provides basic keyword
scoring as a substitute for BM25.
"""

from typing import List


class InMemoryRetriever:
    """Simple in-memory text store with keyword-based search."""

    def __init__(self) -> None:
        self._store: dict = {}  # collection_name -> list[dict]

    # ------------------------------------------------------------------
    # Schema / lifecycle (mirrors WeaviateRetriever interface)
    # ------------------------------------------------------------------

    def create_collection(self, name: str) -> None:
        """Initialise an empty document collection."""
        if name not in self._store:
            self._store[name] = []

    def drop_collection(self, name: str) -> None:
        """Remove a collection and its documents."""
        self._store.pop(name, None)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_documents(self, collection_name: str, sections: list) -> None:
        """Store chunked document sections in memory."""
        try:
            if collection_name not in self._store:
                self._store[collection_name] = []

            for section in sections:
                for doc in section["documents"]:
                    self._store[collection_name].append(
                        {"content": doc.page_content, "tag": section["heading"]}
                    )
        except Exception as exc:
            print(f"[LexAI] In-memory indexing error: {exc}")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def keyword_search(self, query: str, collection_name: str) -> List[str]:
        """Score documents by simple keyword frequency and return top hits."""
        if collection_name not in self._store:
            return []

        tokens = query.lower().split()
        scored = []

        for doc in self._store[collection_name]:
            text_lower = doc["content"].lower()
            score = sum(text_lower.count(tok) for tok in tokens)
            if score > 0:
                scored.append((score, doc["content"]))

        # Highest-scoring chunks first
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [chunk for _, chunk in scored[:4]]
