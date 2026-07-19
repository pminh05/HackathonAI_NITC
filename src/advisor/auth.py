"""Supabase Auth token verification for optional authenticated chat sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import httpx


class InvalidAccessToken(ValueError):
    """The supplied bearer token is missing, expired, or not a user token."""


class AuthenticationServiceUnavailable(RuntimeError):
    """Supabase Auth could not be reached to validate a token."""


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    email: str | None = None

    def as_identity(self) -> dict[str, Any]:
        return {
            "authenticated": True,
            "user_id": self.user_id,
        }


class SupabaseAuthenticator:
    """Verify access tokens through Supabase's authoritative Auth endpoint."""

    def __init__(
        self,
        *,
        supabase_url: str,
        publishable_key: str,
        timeout_seconds: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.supabase_url = supabase_url.rstrip("/")
        self.publishable_key = publishable_key
        self.timeout_seconds = timeout_seconds
        self._client = client
        self._owns_client = client is None

    async def verify(self, access_token: str) -> AuthenticatedUser:
        token = access_token.strip()
        if not token:
            raise InvalidAccessToken("Empty bearer token")
        client = self._client or httpx.AsyncClient()
        if self._client is None:
            self._client = client
        try:
            response = await client.get(
                f"{self.supabase_url}/auth/v1/user",
                headers={
                    "apikey": self.publishable_key,
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                },
                timeout=self.timeout_seconds,
            )
        except httpx.RequestError as exc:
            raise AuthenticationServiceUnavailable(
                "Supabase Auth is temporarily unavailable"
            ) from exc
        if response.status_code in {401, 403}:
            raise InvalidAccessToken("Invalid or expired bearer token")
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise AuthenticationServiceUnavailable(
                "Supabase Auth returned an unexpected response"
            ) from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise AuthenticationServiceUnavailable(
                "Supabase Auth returned invalid JSON"
            ) from exc
        raw_user_id = payload.get("id") or payload.get("sub")
        try:
            user_id = str(UUID(str(raw_user_id)))
        except (TypeError, ValueError, AttributeError) as exc:
            raise InvalidAccessToken("Supabase response did not contain a user UUID") from exc
        email = payload.get("email")
        return AuthenticatedUser(
            user_id=user_id,
            email=str(email) if isinstance(email, str) else None,
        )

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None
