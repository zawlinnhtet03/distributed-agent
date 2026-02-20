"""Aggregation helpers to merge results from multiple shards."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from .models import SearchHit


def normalize_scores(
    results_by_shard: Dict[str, List[SearchHit]],
) -> Dict[str, List[Tuple[SearchHit, float]]]:
    normalized: Dict[str, List[Tuple[SearchHit, float]]] = {}
    for shard, hits in results_by_shard.items():
        if not hits:
            normalized[shard] = []
            continue
        scores = [hit.score for hit in hits]
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            normalized[shard] = [(hit, 1.0) for hit in hits]
            continue
        normalized[shard] = [
            (hit, (hit.score - min_score) / (max_score - min_score)) for hit in hits
        ]
    return normalized


def dedup(
    results: Iterable[Tuple[SearchHit, float]],
    key: str = "payload.global_id",
) -> List[Tuple[SearchHit, float]]:
    selected: Dict[str, Tuple[SearchHit, float]] = {}
    for hit, score in results:
        dedup_key = _extract_key(hit, key)
        existing = selected.get(dedup_key)
        if existing is None or score > existing[1]:
            selected[dedup_key] = (hit, score)
    return list(selected.values())


def merge_topk(results_by_shard: Dict[str, List[SearchHit]], top_k: int) -> List[SearchHit]:
    normalized = normalize_scores(results_by_shard)
    all_hits: List[Tuple[SearchHit, float]] = []
    for shard, hits in normalized.items():
        for hit, norm_score in hits:
            all_hits.append((hit, norm_score))

    deduped = dedup(all_hits)
    deduped.sort(key=lambda item: item[1], reverse=True)
    final_hits = []
    for hit, norm_score in deduped[:top_k]:
        hit.score = norm_score
        final_hits.append(hit)
    return final_hits


def _extract_key(hit: SearchHit, key: str) -> str:
    if key == "payload.global_id":
        if isinstance(hit.payload, dict) and hit.payload.get("global_id"):
            return str(hit.payload["global_id"])
    return str(hit.id)
