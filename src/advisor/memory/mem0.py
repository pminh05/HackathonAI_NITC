"""Mem0 integration placeholder."""

from __future__ import annotations

from typing import Any


class Mem0Memory:
    """Placeholder interface for a future long-term memory integration."""

    def add(self, payload: dict[str, Any]) -> None:
        """Store memory after a future Mem0 integration is specified."""
        raise NotImplementedError("TODO: Implement Mem0 integration.")

    def search(self, query: str) -> list[dict[str, Any]]:
        """Search memory after a future Mem0 integration is specified."""
        raise NotImplementedError("TODO: Implement Mem0 integration.")
