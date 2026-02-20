"""Sharding policy for routing requests to shards."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from .shard_registry import get_registry


def choose_shard(payload: Dict[str, Any], explicit_target: Optional[str] = None) -> str:
    registry = get_registry()
    shard_names = set(registry.list_shards())

    if explicit_target is not None:
        if explicit_target not in shard_names:
            raise ValueError(f"Unknown shard target: {explicit_target}")
        return explicit_target

    content_type = _require_field(payload, "content_type")
    orientation = _require_field(payload, "orientation")
    duration_sec = _require_number(payload, "duration_sec")

    content_type_value = str(content_type).lower()
    orientation_value = str(orientation).lower()

    if content_type_value in {"image", "carousel"}:
        return "shard3"
    if orientation_value == "vertical" and duration_sec <= 90:
        return "shard1"
    return "shard2"


def choose_shard_from_filters(
    filters: Optional[Dict[str, Any]],
    explicit_target: Optional[str] = None,
) -> str:
    registry = get_registry()
    shard_names = set(registry.list_shards())

    if explicit_target is not None:
        if explicit_target not in shard_names:
            raise ValueError(f"Unknown shard target: {explicit_target}")
        return explicit_target

    filters = filters or {}
    content_type_values = _get_filter_values(filters.get("content_type"))
    if content_type_values and set(content_type_values).issubset({"image", "carousel"}):
        return "shard3"

    orientation_values = _get_filter_values(filters.get("orientation"))
    duration_value = filters.get("duration_sec")
    if orientation_values and set(orientation_values) == {"vertical"}:
        if _duration_implies_short(duration_value):
            return "shard1"

    return "shard2"


def _require_field(payload: Dict[str, Any], field_name: str) -> Any:
    if field_name not in payload:
        raise ValueError(f"Missing required field: {field_name}")
    value = payload[field_name]
    if value is None:
        raise ValueError(f"Missing required field: {field_name}")
    return value


def _require_number(payload: Dict[str, Any], field_name: str) -> float:
    value = _require_field(payload, field_name)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Field {field_name} must be numeric") from exc


def _get_filter_values(value: Any) -> Optional[Iterable[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item).lower() for item in value]
    if isinstance(value, str):
        return [value.lower()]
    return None


def _duration_implies_short(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value <= 90
    if isinstance(value, dict):
        if "lte" in value and value["lte"] is not None:
            return value["lte"] <= 90
        if "lt" in value and value["lt"] is not None:
            return value["lt"] <= 90
    return False
