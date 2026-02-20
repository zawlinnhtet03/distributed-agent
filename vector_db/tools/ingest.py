"""CLI tool to ingest JSONL data into the router."""
from __future__ import annotations

import argparse
import json
import time
from typing import Dict, Iterable, List, Optional

import httpx

from router.app.router_policy import choose_shard


def _read_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _post_batch(
    client: httpx.Client,
    router_url: str,
    batch: List[dict],
    target_shard: Optional[str],
    retries: int,
    backoff_s: float,
) -> dict:
    payload = {
        "points": batch,
        "target_shard": target_shard,
        "batch_size": len(batch),
    }

    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            response = client.post(f"{router_url}/upsert", json=payload)
            if response.status_code >= 400:
                raise RuntimeError(f"{response.status_code} {response.text}")
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= retries:
                break
            time.sleep(backoff_s * (attempt + 1))
    raise RuntimeError(f"batch failed after retries: {last_exc}")


def _print_stats(stats: Dict[str, int], total: int, failed: int) -> None:
    shard_lines = ", ".join(f"{name}={count}" for name, count in sorted(stats.items()))
    print(f"processed={total} failed={failed} shard_stats=({shard_lines})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest JSONL data into the vector router.")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--router", default="http://localhost:8000", help="Router base URL")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--target-shard", default=None)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--backoff", type=float, default=0.5)
    args = parser.parse_args()

    shard_stats: Dict[str, int] = {}
    total = 0
    failed = 0

    if args.dry_run:
        for point in _read_jsonl(args.input):
            shard_name = choose_shard(point.get("payload", {}), args.target_shard)
            shard_stats[shard_name] = shard_stats.get(shard_name, 0) + 1
            total += 1
        _print_stats(shard_stats, total, failed)
        return

    batch: List[dict] = []
    with httpx.Client(timeout=30.0) as client:
        for point in _read_jsonl(args.input):
            batch.append(point)
            if len(batch) >= args.batch_size:
                try:
                    response = _post_batch(
                        client,
                        args.router,
                        batch,
                        args.target_shard,
                        args.retries,
                        args.backoff,
                    )
                    for shard, count in response.get("shards_written", {}).items():
                        shard_stats[shard] = shard_stats.get(shard, 0) + count
                except Exception as exc:  # noqa: BLE001
                    failed += len(batch)
                    print(f"batch failed: {exc}")
                total += len(batch)
                _print_stats(shard_stats, total, failed)
                batch = []

        if batch:
            try:
                response = _post_batch(
                    client,
                    args.router,
                    batch,
                    args.target_shard,
                    args.retries,
                    args.backoff,
                )
                for shard, count in response.get("shards_written", {}).items():
                    shard_stats[shard] = shard_stats.get(shard, 0) + count
            except Exception as exc:  # noqa: BLE001
                failed += len(batch)
                print(f"batch failed: {exc}")
            total += len(batch)
            _print_stats(shard_stats, total, failed)


if __name__ == "__main__":
    main()
