"""
Configuration management for the Multi Agent Intelligence Research Hub.

Loads environment variables and provides typed settings.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


_REPO_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=(
            str(_REPO_ROOT / ".env"),
            str(_REPO_ROOT / ".env.local"),
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    mistral_api_key: str | None = None
    tavily_api_key: str

    # Model Configuration (Mistral via LiteLLM)
    # See: https://docs.litellm.ai/docs/providers/mistral
    default_model: str = "mistral/mistral-medium-latest"
    fast_model: str = "mistral/mistral-small-latest"
    embedding_model: str = "mistral/mistral-embed"

    # Model Parameters
    default_temperature: float = 0.7
    default_max_tokens: int = 4096

    llm_num_retries: int = 5
    llm_retry_base_delay_s: float = 1.0
    llm_retry_max_delay_s: float = 20.0

    # Debug
    debug: bool = False

    # RAG Storage (ChromaDB)
    rag_persist_dir: str = "data/chroma"
    rag_collection: str = "default"

    @property
    def model_ids(self) -> dict[str, str]:
        """Available model identifiers for different tiers."""
        return {
            "default": self.default_model,
            "fast": self.fast_model,
            "embedding": self.embedding_model,
        }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Type alias for model tiers
ModelTier = Literal["default", "fast", "embedding"]
