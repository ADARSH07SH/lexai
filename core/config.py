"""
Application configuration and environment variable management.

Loads API keys and service URLs from a .env file using python-dotenv.
All sensitive credentials are kept out of source control.
"""

import os
from dotenv import load_dotenv

# Load .env file at module import time
load_dotenv()


class AppConfig:
    """Centralized configuration for LexAI services."""

    def __init__(self):
        """Load configuration at runtime to ensure Streamlit secrets are available."""
        self.GROQ_API_KEY = self._get_secret("GROQ_API_KEY")
        self.WEAVIATE_API_KEY = self._get_secret("WEAVIATE_API_KEY")
        self.WEAVIATE_URL = self._get_secret("WEAVIATE_URL")

    @staticmethod
    def _get_secret(key: str) -> str:
        """Safely fetch a secret from env vars, falling back to Streamlit secrets."""
        val = os.getenv(key, "")
        if not val:
            try:
                import streamlit as st
                val = st.secrets.get(key, "")
            except Exception:
                pass
        return val

    # Model defaults
    DEFAULT_MODEL = "llama-3.1-8b-instant"
    CONTEXT_CHAR_LIMIT = 15000
    SEARCH_RESULT_LIMIT = 8
