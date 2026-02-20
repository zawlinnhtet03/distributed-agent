from qdrant_client.http import models

from router.app.filters import build_filter


def test_build_filter_match_any_and_range() -> None:
    filters = {
        "platform": ["tiktok", "instagram"],
        "duration_sec": {"lte": 90},
    }
    q_filter = build_filter(filters)
    assert q_filter is not None
    assert isinstance(q_filter, models.Filter)
    assert len(q_filter.must) == 2


def test_build_filter_match_value() -> None:
    filters = {"orientation": "vertical"}
    q_filter = build_filter(filters)
    assert q_filter is not None
    condition = q_filter.must[0]
    assert condition.key == "orientation"
    assert condition.match is not None


def test_build_filter_empty() -> None:
    assert build_filter({}) is None
    assert build_filter(None) is None
