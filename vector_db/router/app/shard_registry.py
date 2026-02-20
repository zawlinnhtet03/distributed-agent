"""Shard registry for routing and health checks."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional

import httpx

from .config import settings


@dataclass(frozen=True)
class ShardInfo:
    name: str
    url: str


class ShardRegistry:
    def __init__(self, shards: List[ShardInfo], timeout_s: float) -> None:
        self._shards = shards
        self._timeout_s = timeout_s
        self._shard_map = {shard.name: shard for shard in shards}
        self._health_cache: Dict[str, bool] = {shard.name: False for shard in shards}

    def list_shards(self) -> List[str]:
        return [shard.name for shard in self._shards]

    def get_url(self, shard_name: str) -> str:
        if shard_name not in self._shard_map:
            raise KeyError(f"Unknown shard: {shard_name}")
        return self._shard_map[shard_name].url

    async def healthcheck_all(self) -> Dict[str, bool]:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            tasks = [self._check_one(client, shard) for shard in self._shards]
            results = await asyncio.gather(*tasks)
        health = {name: ok for name, ok in results}
        self._health_cache = health
        return health

    async def get_healthy_shards(self) -> List[str]:
        health = await self.healthcheck_all()
        return [name for name in self.list_shards() if health.get(name)]

    async def _check_one(self, client: httpx.AsyncClient, shard: ShardInfo) -> tuple[str, bool]:
        try:
            response = await client.get(f"{shard.url}/healthz")
            return shard.name, response.status_code == 200
        except Exception:
            return shard.name, False


_REGISTRY: Optional[ShardRegistry] = None


def get_registry() -> ShardRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ShardRegistry(
            shards=[
                ShardInfo(name="shard1", url=settings.shard1_url),
                ShardInfo(name="shard2", url=settings.shard2_url),
                ShardInfo(name="shard3", url=settings.shard3_url),
            ],
            timeout_s=settings.request_timeout_s,
        )
    return _REGISTRY
