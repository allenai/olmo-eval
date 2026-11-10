"""BrowseComp-Plus HTTP endpoints and upstream result formatting."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from olmo_eval.evals.tasks.browsecomp_plus.data import CORPUS_REPO, EMBEDDING_MODEL
from olmo_eval.evals.tasks.browsecomp_plus.retrieval import server
from olmo_eval.evals.tasks.browsecomp_plus.retrieval.upstream.searcher.tools import register_routes
from olmo_eval.harness.tools.browsecomp_plus import call_retriever


@pytest.fixture
def retriever(monkeypatch):
    tokenizer = Mock()
    tokenizer.encode.side_effect = lambda text, **kwargs: list(text)
    tokenizer.decode.side_effect = lambda tokens, **kwargs: "".join(tokens)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", Mock(return_value=tokenizer))
    return SimpleNamespace(
        search=Mock(return_value=[{"docid": "17", "score": 0.9, "text": "Paris"}]),
        get_document=Mock(return_value={"docid": "17", "text": "document " * 10000}),
        search_description=lambda k: f"Search top-{k} documents",
        get_document_description=lambda: "Get document",
    )


@pytest.mark.anyio
@pytest.mark.parametrize("endpoint", [None, "http://retriever-host:9090/"])
async def test_http_roundtrip_preserves_results(monkeypatch, retriever, endpoint):
    app = FastAPI()
    register_routes(app, retriever, snippet_max_tokens=512, k=5, include_get_document=True)
    original_client = httpx.AsyncClient
    requested_urls = []

    async def record_request(request):
        requested_urls.append(str(request.url))

    def client(**kwargs):
        return original_client(
            transport=httpx.ASGITransport(app=app),
            event_hooks={"request": [record_request]},
            **kwargs,
        )

    monkeypatch.setattr(httpx, "AsyncClient", client)
    if endpoint is None:
        monkeypatch.delenv("BROWSECOMP_PLUS_SEARCH_SERVER_URL", raising=False)
    else:
        monkeypatch.setenv("BROWSECOMP_PLUS_SEARCH_SERVER_URL", endpoint)
    hits = json.loads(await call_retriever("search", {"query": "café in Paris"}))
    assert hits == [{"docid": "17", "score": 0.9, "snippet": "Paris"}]
    retriever.search.assert_called_once_with("café in Paris", 5)
    document = json.loads(await call_retriever("get_document", {"docid": "17"}))
    assert document == {"docid": "17", "text": "document " * 10000}
    retriever.get_document.assert_called_once_with("17")
    retriever.get_document.return_value = None
    assert await call_retriever("get_document", {"docid": "missing"}) == "null"
    retriever.search.return_value = []
    assert await call_retriever("search", {"query": "no hits"}) == "[]"
    base_url = (endpoint or "http://127.0.0.1:8081").rstrip("/")
    assert requested_urls == [
        f"{base_url}/{name}" for name in ("search", "get_document", "get_document", "search")
    ]


@pytest.mark.parametrize("snippet_limit", [512, -1])
def test_snippets_and_optional_scores_follow_upstream(retriever, snippet_limit):
    full_text = "é" * 513
    retriever.search.return_value = [
        {"docid": "1", "score": 0, "text": full_text},
        {"docid": "2", "score": None, "text": "short"},
        {"docid": "3", "text": ""},
        {"docid": "4", "text": "x" * 512},
    ]
    app = FastAPI()
    register_routes(app, retriever, snippet_max_tokens=snippet_limit)
    with TestClient(app) as client:
        response = client.post("/search", json={"query": "Paris"})
    assert response.status_code == 200
    assert response.json() == [
        {"docid": "1", "score": 0, "snippet": full_text[:512] if snippet_limit > 0 else full_text},
        {"docid": "2", "snippet": "short"},
        {"docid": "3", "snippet": ""},
        {"docid": "4", "snippet": "x" * 512},
    ]


@pytest.mark.parametrize("include_get_document", [False, True])
def test_document_endpoint_is_optional_and_inputs_are_validated(retriever, include_get_document):
    app = FastAPI()
    register_routes(app, retriever, include_get_document=include_get_document)
    with TestClient(app) as client:
        assert client.post("/search", json={}).status_code == 422
        assert client.post("/search", json={"query": ["invalid"]}).status_code == 422
        response = client.post("/get_document", json={"docid": "17"})
        assert response.status_code == (200 if include_get_document else 404)
    retriever.search.assert_not_called()
    assert retriever.get_document.call_count == int(include_get_document)


@pytest.fixture
def server_runtime(monkeypatch, retriever):
    def parse_args(parser):
        for name in ("index-path", "model-name", "dataset-name", "pooling", "torch-dtype"):
            parser.add_argument(f"--{name}")
        parser.add_argument("--normalize", action="store_true")
        parser.add_argument("--max-length", type=int)
        parser.add_argument("--task-prefix", default="upstream default")

    searcher_class = Mock(return_value=retriever)
    searcher_class.parse_args.side_effect = parse_args
    get_searcher_class = Mock(return_value=searcher_class)
    searcher_types = SimpleNamespace(get_searcher_class=get_searcher_class)
    modules = {
        "searcher.searchers": SimpleNamespace(SearcherType=searcher_types),
        "searcher.tools": SimpleNamespace(register_routes=register_routes),
    }
    monkeypatch.setattr(server, "import_module", modules.__getitem__)
    monkeypatch.setattr(server, "load_dotenv", Mock())
    monkeypatch.setattr(server.sys, "path", list(server.sys.path))
    run = Mock()
    monkeypatch.setattr(server.uvicorn, "run", run)
    return searcher_class, get_searcher_class, run


@pytest.fixture
def artifacts(tmp_path):
    for index in ("bm25/segments_1", "qwen3-embedding-8b/corpus.shard0.pkl"):
        path = tmp_path / "indexes" / index
        path.parent.mkdir(parents=True)
        path.touch()
    return tmp_path


@pytest.mark.parametrize("backend", ["bm25", "qwen3-embedding-8b"])
@pytest.mark.parametrize("include_get_document", [False, True])
@pytest.mark.parametrize("address", [None, ("0.0.0.0", 9090)])
def test_server_starts_with_reference_settings(
    monkeypatch, retriever, server_runtime, artifacts, backend, include_get_document, address
):
    searcher_class, get_searcher_class, run = server_runtime
    monkeypatch.setenv("BROWSECOMP_PLUS_HOME", "/unused/env/home")
    monkeypatch.setenv("HF_HOME", "/shared/huggingface")
    args = ["server", "--home", str(artifacts), "--retriever", backend]
    if include_get_document:
        args.append("--get-document")
    if address is not None:
        args += ["--host", address[0], "--port", str(address[1])]
    monkeypatch.setattr(server.sys, "argv", args)
    server.main()
    host, port = address or ("127.0.0.1", 8081)
    assert run.call_args.kwargs == {"host": host, "port": port}
    with TestClient(run.call_args.args[0]) as client:
        assert client.post("/search", json={"query": "Paris"}).status_code == 200
        response = client.post("/get_document", json={"docid": "17"})
        assert response.status_code == (200 if include_get_document else 404)
    retriever.search.assert_called_once_with("Paris", 5)
    assert os.environ["HF_HOME"] == "/shared/huggingface"
    configured = searcher_class.call_args.args[0]
    if backend == "bm25":
        get_searcher_class.assert_called_once_with("bm25")
        assert configured.index_path == str(artifacts / "indexes/bm25")
    else:
        get_searcher_class.assert_called_once_with("faiss")
        assert configured.index_path == str(
            artifacts / "indexes/qwen3-embedding-8b/corpus.shard*.pkl"
        )
        assert configured.model_name == EMBEDDING_MODEL
        assert configured.dataset_name == CORPUS_REPO
        assert configured.normalize is True
        assert configured.pooling == "eos"
        assert configured.torch_dtype == "float16"
        assert configured.max_length == 8192
        assert configured.task_prefix == "upstream default"


@pytest.mark.parametrize("home_setting", [None, "artifacts", "~/artifacts"])
def test_server_resolves_default_artifact_directory(
    monkeypatch, tmp_path, retriever, server_runtime, home_setting
):
    monkeypatch.chdir(tmp_path)
    if home_setting is None:
        monkeypatch.delenv("BROWSECOMP_PLUS_HOME", raising=False)
        home = Path.home() / ".cache/olmo-eval/browsecomp-plus"
    else:
        monkeypatch.setenv("BROWSECOMP_PLUS_HOME", home_setting)
        home = (Path.home() if home_setting.startswith("~") else tmp_path) / "artifacts"
    create_searcher = Mock(return_value=retriever)
    monkeypatch.setattr(server, "create_searcher", create_searcher)
    monkeypatch.setattr(server.sys, "argv", ["server", "--retriever", "bm25"])
    server.main()
    create_searcher.assert_called_once_with(home.resolve(), "bm25")


@pytest.mark.parametrize("backend", ["bm25", "qwen3-embedding-8b"])
def test_unprepared_index_fails_before_loading_searcher(monkeypatch, tmp_path, backend):
    load = Mock()
    monkeypatch.setattr(server, "import_module", load)
    with pytest.raises(ValueError, match=f"Prepare the {backend}"):
        server.create_searcher(tmp_path, backend)
    load.assert_not_called()


def test_server_help_works_outside_checkout(tmp_path):
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, server.__file__, "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--retriever {bm25,qwen3-embedding-8b}" in result.stdout
