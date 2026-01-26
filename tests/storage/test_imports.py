"""Test that all storage modules can be imported."""

import importlib

import pytest


def test_import_base():
    """Test importing base storage module."""


def test_import_models():
    """Test importing SQLAlchemy models."""
    try:
        mod = importlib.import_module("olmo_eval.storage.models")
        assert hasattr(mod, "Base")
        assert hasattr(mod, "Experiment")
        assert hasattr(mod, "InstancePrediction")
        assert hasattr(mod, "TaskResult")
    except ImportError as e:
        pytest.skip(f"SQLAlchemy not installed: {e}")


def test_import_session():
    """Test importing session management."""
    try:
        mod = importlib.import_module("olmo_eval.storage.session")
        assert hasattr(mod, "DatabaseSession")
        assert hasattr(mod, "create_postgres_engine")
        assert hasattr(mod, "create_session_factory")
    except ImportError as e:
        pytest.skip(f"SQLAlchemy not installed: {e}")


def test_import_repository():
    """Test importing repository layer."""
    try:
        mod = importlib.import_module("olmo_eval.storage.repository")
        assert hasattr(mod, "ExperimentRepository")
        assert hasattr(mod, "InstancePredictionRepository")
    except ImportError as e:
        pytest.skip(f"SQLAlchemy not installed: {e}")


def test_import_queries():
    """Test importing query helpers."""
    try:
        mod = importlib.import_module("olmo_eval.storage.queries")
        assert hasattr(mod, "QueryHelper")
    except ImportError as e:
        pytest.skip(f"SQLAlchemy not installed: {e}")


def test_import_postgres_backend():
    """Test importing PostgresBackend."""
    try:
        mod = importlib.import_module("olmo_eval.storage.postgres")
        assert hasattr(mod, "PostgresBackend")
    except ImportError as e:
        pytest.skip(f"Required dependencies not installed: {e}")


def test_postgres_backend_has_new_methods():
    """Test that PostgresBackend has the new save_with_instances method."""
    try:
        mod = importlib.import_module("olmo_eval.storage.postgres")
        backend_cls = mod.PostgresBackend

        assert hasattr(backend_cls, "save_with_instances")
        assert hasattr(backend_cls, "save")
        assert hasattr(backend_cls, "get")
        assert hasattr(backend_cls, "query")
        assert hasattr(backend_cls, "delete")
        assert hasattr(backend_cls, "initialize")
        assert hasattr(backend_cls, "cleanup")
        assert hasattr(backend_cls, "dispose")
    except ImportError as e:
        pytest.skip(f"PostgresBackend not available: {e}")
