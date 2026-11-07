"""
SearXNG access shared by every researcher in the process: a result cache, a
throttle, and de-duplication of identical in-flight queries.

Why: upstream engines (Brave, Google, DuckDuckGo, …) rate-limit or suspend a
SearXNG instance that sends them bursts of queries. A five-researcher run used
to fire ~100 searches, many of them near-duplicates, all at once.

  cache      Results for a query are reused for `search_cache_hours` (default
             24) across researchers, runs and restarts (SQLite file under
             .cache/). Empty results and errors are never cached.
  in-flight  Two researchers asking the same query at the same moment share
             one request.
  throttle   Requests to SearXNG start at least `search_min_interval_s`
             (default 1.5 s) apart, process-wide.

Settings live in agents/swarm.toml. Queries are normalized (case, whitespace)
before lookup, so "NVIDIA  Rubin" and "nvidia rubin" are the same entry.
"""

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

CACHE_PATH = Path(__file__).parent / ".cache" / "search.sqlite"


def normalize(query: str) -> str:
    return " ".join(query.lower().split())


class SearchCache:
    """query -> results, with a TTL. SQLite so it survives restarts and is
    safe to share between the CLI and the web server."""

    def __init__(self, path: Path, ttl_s: float) -> None:
        self.ttl_s = ttl_s
        self._db: sqlite3.Connection | None = None
        if ttl_s <= 0:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(path, timeout=5, check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("CREATE TABLE IF NOT EXISTS search "
                             "(key TEXT PRIMARY KEY, stored REAL NOT NULL, results TEXT NOT NULL)")
            self._db.execute("DELETE FROM search WHERE stored < ?", (time.time() - ttl_s,))
            self._db.commit()
        except sqlite3.Error as e:   # a broken cache must never break search
            print(f"[search] cache disabled: {e}")
            self._db = None

    def get(self, key: str) -> list | None:
        if not self._db:
            return None
        try:
            row = self._db.execute("SELECT stored, results FROM search WHERE key = ?", (key,)).fetchone()
        except sqlite3.Error:
            return None
        if not row or row[0] < time.time() - self.ttl_s:
            return None
        return json.loads(row[1])

    def put(self, key: str, results: list) -> None:
        if not self._db or not results:
            return
        try:
            self._db.execute("INSERT OR REPLACE INTO search (key, stored, results) VALUES (?, ?, ?)",
                             (key, time.time(), json.dumps(results)))
            self._db.commit()
        except sqlite3.Error:
            pass


class Throttle:
    """Request start times at least `interval_s` apart, across all callers."""

    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self._next = 0.0
        self._lock: asyncio.Lock | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def wait(self) -> None:
        if self.interval_s <= 0:
            return
        loop = asyncio.get_running_loop()
        if self._lock is None or self._loop is not loop:
            self._lock, self._loop = asyncio.Lock(), loop
        async with self._lock:
            delay = self._next - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self._next = time.monotonic() + self.interval_s


class SharedSearch:
    """Cache + in-flight de-duplication + throttle around a fetch function.

    fetch(query) must return SearXNG's JSON as a dict ({"results": [...],
    "unresponsive_engines": [...]}) or {"error": "..."}; it may raise
    exceptions that should reach the caller (e.g. "service is down")."""

    def __init__(self, cache: SearchCache, throttle: Throttle) -> None:
        self.cache = cache
        self.throttle = throttle
        self._inflight: dict[str, asyncio.Task] = {}

    def _done(self, key: str, task: asyncio.Task) -> None:
        if self._inflight.get(key) is task:
            del self._inflight[key]
        if not task.cancelled():
            task.exception()   # retrieved here so an orphaned failure doesn't warn

    async def search(self, key: str, fetch: Callable[[], Awaitable[dict[str, Any]]]) -> tuple[dict[str, Any], str]:
        """Returns (data, how) where how is "cache", "shared" or "network"."""
        hit = self.cache.get(key)
        if hit is not None:
            return {"results": hit}, "cache"
        task = self._inflight.get(key)
        how = "shared"
        if task is None or task.done():
            how = "network"

            async def run() -> dict[str, Any]:
                await self.throttle.wait()
                return await fetch()

            task = asyncio.create_task(run())
            self._inflight[key] = task
            task.add_done_callback(lambda t, k=key: self._done(k, t))
        # shield: one caller being cancelled (user stopped that researcher)
        # must not cancel the request another researcher is waiting on.
        data = await asyncio.shield(task)
        if how == "network" and data.get("results"):
            self.cache.put(key, data["results"])
        return data, how
