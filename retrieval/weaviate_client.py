"""
Weaviate vector-store integration.

Provides document indexing and BM25 keyword search through a
Weaviate Cloud instance. Used as the primary retriever in the
RAG pipeline.
"""

import uuid
import weaviate
from core.config import AppConfig


class WeaviateRetriever:
    """Manages document storage and retrieval via Weaviate Cloud."""

    def __init__(self) -> None:
        cfg = AppConfig()
        self._api_key = cfg.WEAVIATE_API_KEY
        self._url = cfg.WEAVIATE_URL

        # Ensure the URL has a scheme
        if not self._url.startswith("http"):
            self._url = "https://" + self._url

        self._client = weaviate.Client(
            auth_client_secret=weaviate.AuthApiKey(self._api_key),
            url=self._url,
        )

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    @staticmethod
    def make_collection_name(label: str = None) -> str:
        """Generate a unique Weaviate class name."""
        if label:
            return "Col_" + label
        return "Col_" + str(uuid.uuid4().int)

    def create_collection(self, name: str) -> None:
        """Register a new class/collection in Weaviate."""
        try:
            schema = {
                "class": name,
                "properties": [
                    {"name": "content", "dataType": ["text"]},
                    {"name": "tag", "dataType": ["text"]},
                ],
                "vectorizer": "none",
            }
            self._client.schema.create_class(schema)
        except Exception as exc:
            print(f"[LexAI] Collection creation note: {exc}")

    # ------------------------------------------------------------------
    # Document indexing
    # ------------------------------------------------------------------

    def index_documents(self, collection_name: str, sections: list) -> None:
        """Batch-insert chunked documents into a Weaviate collection."""
        try:
            objects = []
            for section in sections:
                for doc in section["documents"]:
                    objects.append(
                        {"content": doc.page_content, "tag": section["heading"]}
                    )

            self._client.batch.configure(batch_size=100)
            with self._client.batch as batch:
                for obj in objects:
                    batch.add_data_object(obj, collection_name)
        except Exception as exc:
            print(f"[LexAI] Indexing error: {exc}")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def keyword_search(self, query: str, collection_name: str) -> list:
        """Run a BM25 keyword search and return the top matching chunks."""
        response = (
            self._client.query.get(collection_name, ["content", "tag"])
            .with_bm25(query=query)
            .with_limit(AppConfig.SEARCH_RESULT_LIMIT)
            .do()
        )

        if "errors" in response:
            print(f"[LexAI] Search error: {response['errors']}")
            return []

        hits = (
            response.get("data", {})
            .get("Get", {})
            .get(collection_name, [])
        )
        return [hit["content"] for hit in hits] if hits else []

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def drop_collection(self, name: str) -> None:
        """Delete a Weaviate collection and its data."""
        try:
            self._client.schema.delete_class(name)
        except Exception as exc:
            print(f"[LexAI] Drop collection note: {exc}")
