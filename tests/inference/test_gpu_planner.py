"""Tests for GPU planning helpers."""

from __future__ import annotations

import logging

import pytest

from olmo_eval.inference.gpu_planner import resolve_visible_devices


def test_resolve_visible_devices_identity_when_no_outer_mask(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert resolve_visible_devices([0, 1]) == "0,1"
    assert resolve_visible_devices([2, 3]) == "2,3"


def test_resolve_visible_devices_composes_outer_mask(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")

    assert resolve_visible_devices([0, 1]) == "2,3"


def test_resolve_visible_devices_composes_outer_mask_ranges(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")

    assert resolve_visible_devices([0, 1]) == "4,5"
    assert resolve_visible_devices([2, 3]) == "6,7"


def test_resolve_visible_devices_preserves_outer_order(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,2")

    assert resolve_visible_devices([0, 1]) == "3,2"


def test_resolve_visible_devices_tolerates_spaces(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2, 3")

    assert resolve_visible_devices([0, 1]) == "2,3"


@pytest.mark.parametrize("outer", ["", "  "])
def test_resolve_visible_devices_empty_outer_mask_is_unset(monkeypatch, outer):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", outer)

    assert resolve_visible_devices([0, 1]) == "0,1"


def test_resolve_visible_devices_out_of_range_falls_back_with_warning(
    monkeypatch,
    caplog,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")

    with caplog.at_level(logging.WARNING, logger="olmo_eval.inference.gpu_planner"):
        assert resolve_visible_devices([0, 2]) == "4,2"

    assert "GPU index 2 outside CUDA_VISIBLE_DEVICES mask '4,5'; using raw index" in caplog.text
