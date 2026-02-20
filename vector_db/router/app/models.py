"""Pydantic request/response models for the router API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, StrictStr


PointId = Union[int, str]


class NamedVectors(BaseModel):
    frame_clip: List[float]
    text_embed: Optional[List[float]] = None


class UpsertPoint(BaseModel):
    id: StrictStr
    vectors: NamedVectors
    payload: Dict[str, Any]


class UpsertRequest(BaseModel):
    points: List[UpsertPoint]
    target_shard: Optional[StrictStr] = None
    batch_size: int = Field(default=64, ge=1, le=1000)


class SearchRequest(BaseModel):
    mode: StrictStr = Field(default="targeted")
    vector_name: StrictStr
    query_vector: List[float]
    top_k: int = Field(default=10, ge=1, le=1000)
    filters: Optional[Dict[str, Any]] = None
    target_shard: Optional[StrictStr] = None
    with_payload: bool = True
    score_threshold: Optional[float] = None


class SearchHit(BaseModel):
    id: PointId
    score: float
    payload: Optional[Dict[str, Any]] = None
    shard: Optional[str] = None


class SearchResponse(BaseModel):
    mode: str
    shards_queried: List[str]
    took_ms: int
    results: List[SearchHit]
    warnings: List[str]
