"""
Central configuration for the RLAIF RAG Agent.

All values are read from environment variables (or a .env file).
No secrets are ever hard-coded here.

Usage:
    from .settings import settings
    client = openai.OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_api_base)
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Resolve .env relative to this file so it is found regardless of the
# working directory from which Python is launched.
_HERE = Path(__file__).parent
_ENV_FILE = _HERE / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM endpoint ---
    openai_api_key: str = Field(..., description="API key for the LLM endpoint")
    openai_api_base: str = Field(
        "https://llm-server.llmhub.t-systems.net/v2",
        description="Base URL for the OpenAI-compatible endpoint",
    )
    model_name: str = Field(
        "Mixtral-8x7B-Instruct-v0.1-TDU",
        description="Name of the chat model to use",
    )
    embedding_model: str = Field(
        "text-embedding-ada-002",
        description="Name of the OpenAI embedding model",
    )

    # --- LLM generation parameters ---
    temperature_question: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens_question: int = Field(500, gt=0)
    max_tokens_answer: int = Field(2000, gt=0)
    temperature_memory: float = Field(0.0, ge=0.0, le=2.0)

    # --- Retrieval parameters ---
    retrieval_k: int = Field(5, gt=0, description="Number of docs to retrieve per query")
    chunk_size: int = Field(1000, gt=0)
    chunk_overlap: int = Field(200, ge=0)
    max_context_iterations: int = Field(3, gt=0)
    faiss_index_dir: str = Field(
        "faiss_index", description="Directory to persist/load the FAISS index"
    )

    # --- Memory parameters ---
    short_memory_max: int = Field(3, gt=0)
    long_memory_max: int = Field(70, gt=0)
    memory_file: str = Field("conversation_history.json")

    # --- Service ---
    service_host: str = Field("0.0.0.0")
    service_port: int = Field(8000, gt=0)
    log_level: str = Field("INFO")


# Module-level singleton — loaded once, available everywhere via import.
settings = Settings()
