"""Event-loop factories used by the API server."""

from __future__ import annotations

import asyncio


def postgres_compatible_loop_factory() -> asyncio.AbstractEventLoop:
    """Return an event loop supported by psycopg's async implementation."""
    return asyncio.SelectorEventLoop()
