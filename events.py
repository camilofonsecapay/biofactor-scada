"""
BioFactor — Bus de eventos in-process (pub/sub) para SSE.
El kernel publica; los clientes del dashboard se suscriben vía /api/stream.
Módulo sin dependencias internas para evitar imports circulares.
"""

import asyncio
import json
from datetime import datetime

_subscribers: "set[asyncio.Queue]" = set()
# Buffer corto del feed reciente para que un cliente que entra vea contexto.
_recent: "list[str]" = []
_RECENT_MAX = 40


def subscribe() -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    _subscribers.add(q)
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    _subscribers.discard(q)


def recent() -> "list[str]":
    return list(_recent)


async def publish(event_type: str, data: dict) -> None:
    payload = {"type": event_type, "ts": datetime.utcnow().isoformat(), "data": data}
    msg = json.dumps(payload, ensure_ascii=False, default=str)
    _recent.append(msg)
    if len(_recent) > _RECENT_MAX:
        del _recent[0]
    for q in list(_subscribers):
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass
