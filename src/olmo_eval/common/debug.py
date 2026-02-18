"""Debug logging utilities for olmo-eval.

Provides centralized debug configuration and logging helpers.
Enable with environment variables:
- OLMO_EVAL_DEBUG_REQUESTS=1: Log HTTP requests/responses
- OLMO_EVAL_DEBUG_VLLM=1: Enable verbose vLLM logging
"""

import json
import logging
import os

import httpx

logger = logging.getLogger(__name__)


def is_debug_requests() -> bool:
    """Check if HTTP request debugging is enabled."""
    return os.getenv("OLMO_EVAL_DEBUG_REQUESTS", "").lower() in ("1", "true", "yes")


def is_debug_provider() -> bool:
    """Check if provider debugging is enabled."""
    return os.getenv("OLMO_EVAL_DEBUG_PROVIDER", "").lower() in ("1", "true", "yes")


async def _log_request(request: httpx.Request) -> None:
    """Log outgoing HTTP request."""
    if not is_debug_requests():
        return

    logger.info(f"Request: {request.method} {request.url}")
    if request.content:
        try:
            body = json.loads(request.content.decode())
            logger.info(f"Body:\n{json.dumps(body, indent=2)}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.info(f"Body: {request.content.decode()}")


async def _log_response(response: httpx.Response) -> None:
    """Log HTTP response."""
    if not is_debug_requests():
        return

    status = response.status_code
    logger.info(f"Response: {status} {response.reason_phrase}")
    try:
        body = await response.aread()
        if body:
            try:
                parsed = json.loads(body.decode())
                logger.info(f"Response body:\n{json.dumps(parsed, indent=2)}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.info(f"Response body: {body.decode()[:1000]}")
    except Exception as e:
        logger.info(f"Could not read response body: {e}")


def create_debug_http_client() -> httpx.AsyncClient:
    """Create an httpx AsyncClient with debug logging enabled."""
    return httpx.AsyncClient(event_hooks={"request": [_log_request], "response": [_log_response]})
