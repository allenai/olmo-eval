"""Cache for compute-heavy responses (spec 4.5).

The key covers the endpoint, the normalized request and ``(task_result_id, updated_at)`` of
every task result used, so re-uploading a task result changes the key. Entries live in
``stats_cache`` (pruned after 30 days at startup) with a small in-process LRU in front.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from olmo_eval_api.services.derive import canonical_json, sha256_hex

_LRU: OrderedDict[str, dict[str, Any]] = OrderedDict()
_LRU_MAX = 512


def cache_key(endpoint: str, request: Any, trs: Iterable[Mapping[Any, Any]]) -> str:
    used = sorted({(int(tr["id"]), tr["updated_at"].timestamp()) for tr in trs})
    payload = {"endpoint": endpoint, "request": request, "trs": used}
    return sha256_hex(canonical_json(payload))


async def get_cached[M: BaseModel](session: AsyncSession, key: str, model: type[M]) -> M | None:
    value = _LRU.get(key)
    if value is None:
        row = (
            await session.execute(text("SELECT value FROM stats_cache WHERE key = :k"), {"k": key})
        ).first()
        if row is None:
            return None
        value = row[0]
        _remember(key, value)
    else:
        _LRU.move_to_end(key)
    return model.model_validate(value)


async def put_cached(session: AsyncSession, key: str, response: BaseModel) -> None:
    value = response.model_dump(mode="json", by_alias=True)
    _remember(key, value)
    await session.execute(
        text(
            "INSERT INTO stats_cache (key, value, created_at) "
            "VALUES (:k, CAST(:v AS jsonb), now()) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value, created_at = now()"
        ),
        {"k": key, "v": canonical_json(value).decode()},
    )
    await session.commit()


def _remember(key: str, value: dict[str, Any]) -> None:
    _LRU[key] = value
    _LRU.move_to_end(key)
    while len(_LRU) > _LRU_MAX:
        _LRU.popitem(last=False)


def clear_memory_cache() -> None:
    _LRU.clear()
