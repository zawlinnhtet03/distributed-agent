"""Configuration loading for the router service."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid int for {name}: {value}") from exc


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid float for {name}: {value}") from exc


def _get_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _get_optional_str(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    return value


@dataclass(frozen=True)
class Settings:
    router_port: int
    shard1_url: str
    shard2_url: str
    shard3_url: str
    qdrant_api_key: Optional[str]
    request_timeout_s: float
    retry_count: int
    retry_backoff_s: float

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            router_port=_get_int("ROUTER_PORT", 8000),
            shard1_url=_get_str("SHARD1_URL", "http://qdrant_shard1:6333"),
            shard2_url=_get_str("SHARD2_URL", "http://qdrant_shard2:6333"),
            shard3_url=_get_str("SHARD3_URL", "http://qdrant_shard3:6333"),
            qdrant_api_key=_get_optional_str("QDRANT_API_KEY"),
            request_timeout_s=_get_float("REQUEST_TIMEOUT_S", 5.0),
            retry_count=_get_int("RETRY_COUNT", 2),
            retry_backoff_s=_get_float("RETRY_BACKOFF_S", 0.5),
        )


settings = Settings.load()
