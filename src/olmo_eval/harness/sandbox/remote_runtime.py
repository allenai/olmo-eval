"""Reliable SWE-ReX HTTP transport for high-concurrency remote sandboxes."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

import aiohttp
from pydantic import BaseModel
from swerex.runtime.abstract import CloseResponse
from swerex.runtime.remote import RemoteRuntime


class ReliableRemoteRuntime(RemoteRuntime):
    """Remote runtime with isolated connections and idempotent transport retries."""

    handles_transport_retries = True

    def __init__(
        self,
        *,
        max_connections: int,
        transport_retries: int = 3,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(logger=logger, **kwargs)
        # Retained for constructor compatibility with the deployment adapter.
        # Do not pool connections: Modal/SWE-ReX closes idle connections and
        # reusing them causes synchronized ServerDisconnectedError bursts.
        del max_connections
        self._transport_retries = max(0, transport_retries)

    def _new_session(self) -> aiohttp.ClientSession:
        """Create an isolated session for one transport attempt."""
        return aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(force_close=True),
            timeout=aiohttp.ClientTimeout(total=self._config.timeout),
        )

    async def _request(
        self,
        endpoint: str,
        payload: BaseModel | None,
        output_class: Any,
        num_retries: int | None = None,
    ) -> Any:
        """Send one logical request, retaining its request ID across retries."""
        request_url = f"{self._api_url}/{endpoint}"
        request_id = str(uuid.uuid4())
        headers = self._headers.copy()
        headers["X-Request-ID"] = request_id
        retries = self._transport_retries if num_retries is None else num_retries
        max_attempts = retries + 1

        for attempt in range(1, max_attempts + 1):
            try:
                async with (
                    self._new_session() as session,
                    session.post(
                        request_url,
                        json=payload.model_dump() if payload else None,
                        headers=headers,
                    ) as response,
                ):
                    await self._handle_response_errors(response)
                    body = await response.json()
                    if attempt > 1:
                        self.logger.info(
                            "Recovered request %s to %s after %d attempts",
                            request_id,
                            endpoint,
                            attempt,
                        )
                    return output_class(**body)
            except (
                aiohttp.ClientError,
                TimeoutError,
                ConnectionError,
                json.JSONDecodeError,
            ) as exc:
                if attempt >= max_attempts:
                    self.logger.error(
                        "Request %s to %s failed after %d attempts: %s",
                        request_id,
                        endpoint,
                        attempt,
                        exc,
                    )
                    raise
                delay = 0.25 * (2 ** (attempt - 1))
                self.logger.warning(
                    "Transport failure for request %s to %s (attempt %d/%d): %s; retrying in %.2fs",
                    request_id,
                    endpoint,
                    attempt,
                    max_attempts,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        raise AssertionError("transport retry loop exited unexpectedly")

    async def close(self) -> CloseResponse:
        try:
            return await self._request("close", None, CloseResponse, num_retries=0)
        except (aiohttp.ClientError, TimeoutError, ConnectionError, json.JSONDecodeError):
            # The close endpoint may terminate the server before its response
            # reaches the client. Shutdown is best-effort and must never leak.
            return CloseResponse()
