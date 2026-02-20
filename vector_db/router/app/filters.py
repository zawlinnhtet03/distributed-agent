"""Translate API filters into Qdrant filter objects."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from qdrant_client.http import models


def build_filter(filters: Optional[Dict[str, Any]]) -> Optional[models.Filter]:
    if not filters:
        return None

    conditions: list[models.FieldCondition] = []
    for field_name, value in filters.items():
        condition = _field_condition(field_name, value)
        if condition is not None:
            conditions.append(condition)

    if not conditions:
        return None

    return models.Filter(must=conditions)


def _field_condition(field_name: str, value: Any) -> Optional[models.FieldCondition]:
    if value is None:
        return None

    if isinstance(value, list):
        return models.FieldCondition(
            key=field_name,
            match=models.MatchAny(any=value),
        )

    if isinstance(value, dict):
        range_params = _range_from_dict(value)
        if range_params is None:
            return None
        return models.FieldCondition(key=field_name, range=range_params)

    return models.FieldCondition(key=field_name, match=models.MatchValue(value=value))


def _range_from_dict(value: Dict[str, Any]) -> Optional[models.Range]:
    supported_keys = {"gte", "lte", "gt", "lt"}
    if not any(key in value for key in supported_keys):
        return None

    range_kwargs: Dict[str, Any] = {}
    for key in supported_keys:
        if key in value:
            range_kwargs[key] = value[key]

    return models.Range(**range_kwargs)
