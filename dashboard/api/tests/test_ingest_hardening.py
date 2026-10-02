"""Ingest edge cases: values that do not fit a column, and concurrent URL signing."""

from __future__ import annotations

import asyncio
import base64
import hashlib
from typing import Any

import httpx

from tests.factories import instance_rows, run_payload, task_payload

RUN_ID = "hard00000001"


async def _run_with_task(client: httpx.AsyncClient) -> int:
    body = run_payload(run_id=RUN_ID, model_name="org/model")
    assert (await client.put(f"/v1/runs/{RUN_ID}", json=body)).status_code == 200
    tr = await client.post(
        f"/v1/runs/{RUN_ID}/task-results",
        json=task_payload(
            "arc", task_hash="h1", primary_metric="acc:default", corpus_score=0.5, n=2
        ),
    )
    assert tr.status_code == 200, tr.text
    return tr.json()["task_result_id"]


async def test_out_of_range_integer_is_a_client_error(client: httpx.AsyncClient) -> None:
    # A 500 would make the upload client retry six times; the request can never succeed.
    tid = await _run_with_task(client)
    rows = instance_rows([1.0, 0.0], "acc:default")
    rows[0]["completion_tokens"] = 2**40  # instance_results.completion_tokens is an integer
    response = await client.post(
        f"/v1/task-results/{tid}/instances", json={"batch_index": 0, "instances": rows}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "bad_request"
    rows[0]["completion_tokens"] = 12
    ok = await client.post(
        f"/v1/task-results/{tid}/instances", json={"batch_index": 0, "instances": rows}
    )
    assert ok.status_code == 200 and ok.json()["total_stored"] == 2


async def test_signing_runs_concurrently_and_keeps_order(
    client: httpx.AsyncClient, app: Any
) -> None:
    await _run_with_task(client)
    storage = app.state.storage
    data = b"already here\n"
    md5 = base64.b64encode(hashlib.md5(data).digest()).decode()
    storage.write(f"dev/runs/{RUN_ID}/present.txt", data)
    original = storage.sign_upload
    active = {"now": 0, "max": 0}

    async def slow_sign(key: str, content_type: str, md5_b64: str) -> Any:
        active["now"] += 1
        active["max"] = max(active["max"], active["now"])
        await asyncio.sleep(0.01)
        active["now"] -= 1
        return await original(key, content_type, md5_b64)

    storage.sign_upload = slow_sign
    artifacts = [
        {
            "path": f"logs/part-{i}.txt",
            "size_bytes": 3,
            "md5_b64": md5,
            "content_type": "text/plain",
            "kind": "logs",
        }
        for i in range(20)
    ]
    artifacts.insert(
        5,
        {
            "path": "present.txt",
            "size_bytes": len(data),
            "md5_b64": md5,
            "content_type": "text/plain",
            "kind": "other",
        },
    )
    response = await client.post(f"/v1/runs/{RUN_ID}/artifacts:sign", json={"artifacts": artifacts})
    assert response.status_code == 200, response.text
    uploads = response.json()["uploads"]
    assert [u["path"] for u in uploads] == [a["path"] for a in artifacts]
    assert [u["skip"] for u in uploads] == [a["path"] == "present.txt" for a in artifacts]
    assert all(u["url"].endswith(u["path"]) for u in uploads if not u["skip"])
    assert active["max"] > 1
