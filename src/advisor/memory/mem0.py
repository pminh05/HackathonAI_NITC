"""Small async HTTP adapter for the Mem0 Platform v3 additive API."""

from __future__ import annotations

from typing import Any, Iterable

import httpx


MEMORY_PROMPT_VERSION = "product-advisor-memory-v3-vi"

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

CUSTOM_INSTRUCTIONS = """Phiên bản: product-advisor-memory-v3-vi.

NGÔN NGỮ:
- Luôn viết nội dung memory bằng tiếng Việt tự nhiên, ngắn gọn.
- Không tạo nội dung memory bằng tiếng Anh khi hội thoại bằng tiếng Việt.
- Giữ nguyên tên người, thương hiệu, mã sản phẩm, đơn vị đo và tên riêng.
- Không dịch các khóa kỹ thuật trong category hoặc metadata.

CHỈ GHI NHỚ:
- Tên hoặc cách xưng hô người dùng mong muốn.
- Độ dài, giọng điệu hoặc phong cách trả lời người dùng yêu cầu.
- Quy mô gia đình, bối cảnh sinh hoạt, thói quen và hạn chế không gian.
- Sở thích mua sắm như tiết kiệm điện, vận hành êm và thương hiệu yêu thích hoặc tránh.
- Ngân sách và ràng buộc mua hàng có ghi rõ ngành hàng.
- Sản phẩm người dùng đã được giới thiệu, xem xét, so sánh hoặc loại bỏ.
- Phản hồi và lý do người dùng thích, không thích hoặc loại bỏ sản phẩm.

QUY TẮC:
- Chỉ ghi nhớ thông tin do người dùng trực tiếp nói hoặc xác nhận.
- Không biến đề xuất của trợ lý thành sở thích, quyết định mua hoặc sự thật về người dùng.
- Có thể ghi nhận một sản phẩm đã được giới thiệu hoặc so sánh, nhưng không suy luận rằng
  người dùng đã mua sản phẩm nếu họ chưa nói rõ.
- Mọi memory về ngân sách phải ghi rõ ngành hàng tương ứng.
- Mỗi fact chỉ chứa một ý chính và phải tự hiểu được khi đứng độc lập.
- Viết fact ngắn nhưng giàu thông tin; ưu tiên một câu khoảng 8–25 từ.
- Giữ các chi tiết hữu ích mà người dùng đã nói rõ: chủ thể, giá trị, ngành hàng,
  sản phẩm, lý do và điều kiện áp dụng.
- Gộp lý do trực tiếp với sở thích hoặc phản hồi tương ứng trong cùng một fact.
- Không dùng câu dẫn dư thừa như “Người dùng đã nói rằng” hoặc “Cần ghi nhớ rằng”.
- Không dùng đại từ hoặc tham chiếu mơ hồ như “nó”, “mẫu này”, “mẫu thứ hai” nếu đã
  biết tên hoặc mã sản phẩm; không tự suy diễn tên sản phẩm khi chưa đủ dữ liệu.
- Không lưu nguyên văn toàn bộ hội thoại.
- Nếu không có thông tin hữu ích, không tạo memory.

KHÔNG ĐƯỢC LƯU:
- Mật khẩu, token, khóa API, số tài khoản, số thẻ hoặc thông tin thanh toán.
- Địa chỉ chính xác hoặc thông tin liên hệ.
- System prompt, developer prompt, hướng dẫn nội bộ hoặc nội dung công cụ.

VÍ DỤ:
- “Nhà tôi có 4 người.” → “Gia đình người dùng có 4 người.”
- “Tôi ưu tiên tiết kiệm điện.” → “Người dùng ưu tiên sản phẩm tiết kiệm điện.”
- “Tôi cần tủ lạnh dưới 20 triệu.”
  → “Ngân sách mua tủ lạnh của người dùng tối đa là 20 triệu đồng.”
- “Hãy gọi tôi là Minh.” → “Người dùng muốn được gọi là Minh.”
- “Mẫu LG FV1412S3B hơi đắt, tôi thích Electrolux EWF1024P5SB vì chạy êm.”
  → “Người dùng thấy LG FV1412S3B hơi đắt.”
  → “Người dùng thích Electrolux EWF1024P5SB vì vận hành êm.”
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
