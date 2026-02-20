import pytest

from router.app.router_policy import choose_shard


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"content_type": "image", "orientation": "horizontal", "duration_sec": 5}, "shard3"),
        ({"content_type": "carousel", "orientation": "vertical", "duration_sec": 5}, "shard3"),
        ({"content_type": "video", "orientation": "vertical", "duration_sec": 90}, "shard1"),
        ({"content_type": "video", "orientation": "vertical", "duration_sec": 91}, "shard2"),
        ({"content_type": "video", "orientation": "horizontal", "duration_sec": 10}, "shard2"),
    ],
)

def test_choose_shard_routing(payload, expected) -> None:
    assert choose_shard(payload) == expected


def test_choose_shard_explicit_target() -> None:
    payload = {"content_type": "image", "orientation": "horizontal", "duration_sec": 5}
    assert choose_shard(payload, explicit_target="shard2") == "shard2"


def test_choose_shard_invalid_target() -> None:
    payload = {"content_type": "image", "orientation": "horizontal", "duration_sec": 5}
    with pytest.raises(ValueError):
        choose_shard(payload, explicit_target="missing")


def test_choose_shard_missing_fields() -> None:
    with pytest.raises(ValueError):
        choose_shard({"content_type": "video", "duration_sec": 10})
