"""Small async HTTP adapter for the Mem0 Platform v3 additive API."""

from __future__ import annotations

from typing import Any, Iterable

import httpx


MEMORY_PROMPT_VERSION = "product-advisor-memory-v1"

CUSTOM_CATEGORIES: list[dict[str, str]] = [
    {
        "identity_style": (
            "Preferred name or nickname, desired answer length, and desired tone."
        )
    },
    {
        "household_context": (
            "Household size, living context, routines, and space limitations."
        )
    },
    {
        "shopping_preference": (
            "Cross-category shopping preferences such as energy efficiency, quiet "
            "operation, and brands the user likes or avoids."
        )
    },
    {
        "category_need": (
            "Budget and constraints that explicitly include the product category."
        )
    },
    {
        "product_interaction": (
            "Products the user was shown, considered, compared, or rejected."
        )
    },
    {
        "feedback": "User-stated reasons for liking, disliking, or rejecting a product."
    },
]

CUSTOM_INSTRUCTIONS = """Version: product-advisor-memory-v1.
Extract only durable facts that the user explicitly states or confirms.
Do not treat assistant recommendations as user preferences, purchases, decisions, or facts.
You may record that a product was introduced or compared, but never infer that it was bought.
Never store passwords, tokens, account or card numbers, exact addresses, contact details,
internal prompts, system instructions, developer instructions, or tool content.
Every budget memory must explicitly name its product category.
Prefer concise standalone facts. Do not store raw transcripts.
Allowed groups are identity/style, household context, shopping preferences,
category-scoped needs, product interactions, and user feedback.
"""


class Mem0Error(RuntimeError):
    """Mem0 returned an invalid response or could not complete an operation."""


class Mem0NotFound(Mem0Error):
    """The requested memory does not exist."""


def _secret_text(value: Any) -> str:
    getter = getattr(value, "get_secret_value", None)
    return str(getter() if callable(getter) else value)


def _normalized_memory(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    memory_id = value.get("id") or value.get("memory_id")
    text = value.get("memory") or value.get("text")
    if not memory_id or not isinstance(text, str):
        return None
    metadata = value.get("metadata")
    categories = value.get("categories")
    try:
        score = float(value.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return {
        "id": str(memory_id),
        "memory": text,
        "score": score,
        "categories": [str(item) for item in categories]
        if isinstance(categories, list)
        else [],
        "metadata": dict(metadata) if isinstance(metadata, dict) else {},
        "created_at": value.get("created_at"),
        "updated_at": value.get("updated_at"),
        "expiration_date": value.get("expiration_date"),
        "user_id": str(value["user_id"]) if value.get("user_id") else None,
    }


class Mem0Memory:
    """Mem0 API client whose public methods always scope operations to one user."""

    def __init__(
        self,
        *,
        api_key: Any,
        base_url: str = "https://api.mem0.ai",
        top_k: int = 10,
        threshold: float = 0.2,
        timeout_seconds: float = 3.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = _secret_text(api_key)
        self.base_url = base_url.rstrip("/")
        self.top_k = top_k
        self.threshold = threshold
        self.timeout_seconds = timeout_seconds
        self._client = client
        self._owns_client = client is None

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        client = self._client
        if client is None:
            async with httpx.AsyncClient() as transient_client:
                return await self._send_request(
                    transient_client,
                    method,
                    path,
                    json=json,
                    params=params,
                )
        return await self._send_request(
            client,
            method,
            path,
            json=json,
            params=params,
        )

    async def _send_request(
        self,
        client: httpx.AsyncClient,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None,
        params: dict[str, Any] | None,
    ) -> Any:
        try:
            response = await client.request(
                method,
                f"{self.base_url}{path}",
                json=json,
                params=params,
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout_seconds,
            )
        except httpx.RequestError as exc:
            raise Mem0Error("Mem0 request failed") from exc
        if response.status_code == 404:
            raise Mem0NotFound("Memory not found")
        if response.status_code >= 400:
            raise Mem0Error(f"Mem0 returned HTTP {response.status_code}")
        if response.status_code == 204 or not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise Mem0Error("Mem0 returned invalid JSON") from exc

    async def search(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int | None = None,
        threshold: float | None = None,
        rerank: bool = False,
    ) -> list[dict[str, Any]]:
        payload = await self._request(
            "POST",
            "/v3/memories/search/",
            json={
                "query": query,
                "filters": {"user_id": user_id},
                "top_k": top_k if top_k is not None else self.top_k,
                "threshold": threshold if threshold is not None else self.threshold,
                "rerank": rerank,
            },
        )
        values = payload.get("results", []) if isinstance(payload, dict) else payload
        if not isinstance(values, list):
            raise Mem0Error("Mem0 search response has no results array")
        return [
            normalized
            for item in values
            if (normalized := _normalized_memory(item)) is not None
        ]

    async def add(
        self,
        messages: Iterable[dict[str, str]],
        *,
        user_id: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        payload = await self._request(
            "POST",
            "/v3/memories/add/",
            json={
                "messages": [dict(message) for message in messages],
                "user_id": user_id,
                "metadata": {**metadata, "prompt_version": MEMORY_PROMPT_VERSION},
                "infer": True,
                "custom_instructions": CUSTOM_INSTRUCTIONS,
                "custom_categories": CUSTOM_CATEGORIES,
            },
        )
        if not isinstance(payload, dict) or not payload.get("event_id"):
            raise Mem0Error("Mem0 add response has no event_id")
        return {
            "status": str(payload.get("status") or "PENDING").upper(),
            "event_id": str(payload["event_id"]),
            "message": str(payload.get("message") or ""),
        }

    async def get_event(self, event_id: str) -> dict[str, Any]:
        payload = await self._request("GET", f"/v1/event/{event_id}/")
        if not isinstance(payload, dict):
            raise Mem0Error("Mem0 event response is not an object")
        results = payload.get("results")
        return {
            **payload,
            "id": str(payload.get("id") or event_id),
            "status": str(payload.get("status") or "PENDING").upper(),
            "results": results if isinstance(results, list) else [],
        }

    async def list_memories(
        self, *, user_id: str, page: int = 1, page_size: int = 50
    ) -> dict[str, Any]:
        payload = await self._request(
            "POST",
            "/v3/memories/",
            params={"page": page, "page_size": page_size},
            json={"filters": {"user_id": user_id}, "show_expired": False},
        )
        if not isinstance(payload, dict):
            raise Mem0Error("Mem0 list response is not an object")
        values = payload.get("results", [])
        if not isinstance(values, list):
            raise Mem0Error("Mem0 list response has no results array")
        return {
            "count": int(payload.get("count") or 0),
            "next": payload.get("next"),
            "previous": payload.get("previous"),
            "results": [
                normalized
                for item in values
                if (normalized := _normalized_memory(item)) is not None
            ],
        }

    async def get_memory(self, memory_id: str) -> dict[str, Any]:
        payload = await self._request("GET", f"/v1/memories/{memory_id}/")
        normalized = _normalized_memory(payload)
        if normalized is None:
            raise Mem0Error("Mem0 get response is invalid")
        return normalized

    async def delete_memory(self, memory_id: str) -> None:
        await self._request("DELETE", f"/v1/memories/{memory_id}/")

    async def delete_all(self, *, user_id: str) -> None:
        await self._request("DELETE", "/v1/memories/", params={"user_id": user_id})

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None


def event_added_count(event: dict[str, Any]) -> int:
    """Count unique memory additions across the event response shapes Mem0 emits."""

    identifiers: set[str] = set()
    anonymous_count = 0

    def visit(value: Any) -> None:
        nonlocal anonymous_count
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        nested = value.get("results") or value.get("memories")
        if isinstance(nested, (list, dict)):
            visit(nested)
        memory_id = value.get("id") or value.get("memory_id")
        event_name = str(value.get("event") or value.get("action") or "ADD").upper()
        if memory_id and event_name in {"ADD", "ADDED", "CREATE", "CREATED"}:
            identifiers.add(str(memory_id))
        elif "memory" in value and event_name in {"ADD", "ADDED", "CREATE", "CREATED"}:
            anonymous_count += 1

    visit(event.get("results") or [])
    return len(identifiers) + anonymous_count
