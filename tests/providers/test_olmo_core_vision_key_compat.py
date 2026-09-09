"""Checkpoints written before and after the ``vision_backbone.`` rename must both load.

OLMo-core moved ``MultimodalLM``'s vision encoder and connector under a ``VisionBackbone``
submodule, so a freshly built model reports ``vision_backbone.vision.*`` while every
checkpoint already on disk has bare ``vision.*``. Back-compat used to be a
``state_dict()``/``load_state_dict()`` override pair on the model, but that rewrote only
*model* keys — optimizer FQNs come from ``named_parameters()`` and were left alone, so one
checkpoint carried two key spaces. The overrides are gone and the remap happens at the
checkpoint layer instead; these tests pin that ``load_checkpoint_weights`` does it.

Synthetic throughout: fake models and tiny safetensors files, so this runs on CPU with no
weka reads and no real checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import olmo_eval.inference.providers.olmo_core_vlm_utils as vlm_utils
from olmo_eval.inference.providers.olmo_core_vlm_utils import (
    MultimodalCheckpointInfo,
    load_checkpoint_weights,
)

torch = pytest.importorskip("torch")

#: What ``MultimodalLM.legacy_vision_key_mapping()`` returns: ``{current: original}``,
#: already the shape ``swap_param_keys`` expects, so no inversion at the call site.
_LEGACY_MAPPING = {
    "vision_backbone.vision.class_embedding": "vision.class_embedding",
    "vision_backbone.connector.w1.weight": "connector.w1.weight",
}


class _FakeModel:
    """Stands in for a built ``MultimodalLM``.

    Only needs what ``load_checkpoint_weights`` touches: ``.lm`` for ``_untie_lm_head``
    (untied here, so it is a no-op) and a recording ``load_state_dict``.
    """

    def __init__(self, *, with_mapping: bool = True) -> None:
        self.lm = SimpleNamespace(tie_word_embeddings=False)
        self.loaded: dict[str, Any] | None = None
        if with_mapping:
            self.legacy_vision_key_mapping = lambda: dict(_LEGACY_MAPPING)

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.loaded = dict(state_dict)


def _info(fmt: str) -> MultimodalCheckpointInfo:
    return MultimodalCheckpointInfo(format=fmt, config={}, model_config={})


# ---------------------------------------------------------------------------
# Path 1: olmo_core_dcp — the primary path for evaluating training checkpoints
# ---------------------------------------------------------------------------


def test_dcp_load_passes_the_legacy_key_mapping(monkeypatch, tmp_path: Path) -> None:
    # Without a key_mapping the loader looks for `vision_backbone.*` in a checkpoint that
    # only has `vision.*`, which is every checkpoint written before the rename.
    captured: dict[str, Any] = {}

    def fake_load(dir_, model, optim=None, **kwargs):
        captured["dir"] = dir_
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "olmo_core.distributed.checkpoint.load_model_and_optim_state", fake_load, raising=True
    )

    model = _FakeModel()
    load_checkpoint_weights(_info("olmo_core_dcp"), str(tmp_path), model)

    assert captured["kwargs"]["key_mapping"] == _LEGACY_MAPPING
    assert captured["dir"].endswith("model_and_optim")


def test_dcp_load_tolerates_a_model_without_the_helper(monkeypatch, tmp_path: Path) -> None:
    # legacy_vision_key_mapping only exists on MultimodalLM; the hasattr guard keeps this
    # path working for anything else that reaches it.
    captured: dict[str, Any] = {}

    def fake_load(dir_, model, optim=None, **kwargs):
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "olmo_core.distributed.checkpoint.load_model_and_optim_state", fake_load, raising=True
    )

    load_checkpoint_weights(_info("olmo_core_dcp"), str(tmp_path), _FakeModel(with_mapping=False))

    assert captured["kwargs"]["key_mapping"] is None


def test_legacy_mapping_is_current_to_original_not_inverted() -> None:
    # swap_param_keys reads {current_key: original_key} and skips entries whose *original*
    # side is absent from the checkpoint metadata. Inverting it would silently no-op on a
    # legacy checkpoint and mis-map a current one.
    for current, original in _LEGACY_MAPPING.items():
        assert current.startswith("vision_backbone.")
        assert not original.startswith("vision_backbone.")
        assert current == f"vision_backbone.{original}"


# ---------------------------------------------------------------------------
# Path 2: olmo_core_unsharded — consolidated exports from scripts/unshard.py,
# which is key-agnostic, so both layouts exist in the wild.
# ---------------------------------------------------------------------------


def _write_export(tmp_path: Path, state_dict: dict[str, Any]) -> Path:
    from safetensors.torch import save_file

    out = tmp_path / "export"
    out.mkdir()
    save_file(state_dict, str(out / "model.safetensors"))
    return out


@pytest.mark.parametrize(
    "vision_key",
    [
        pytest.param("vision.class_embedding", id="legacy-bare-vision"),
        pytest.param("vision_backbone.vision.class_embedding", id="current-vision-backbone"),
    ],
)
def test_unsharded_export_loads_under_either_layout(tmp_path: Path, vision_key: str) -> None:
    # load_state_dict here is strict, so a legacy export used to raise on the whole ViT and
    # connector rather than loading them.
    checkpoint = _write_export(
        tmp_path,
        {
            "lm.embeddings.weight": torch.zeros(2, 2),
            vision_key: torch.zeros(2),
        },
    )

    model = _FakeModel()
    load_checkpoint_weights(_info("olmo_core_unsharded"), str(checkpoint), model)

    assert model.loaded is not None
    # Both layouts arrive canonical, so the model never sees the legacy names.
    assert "vision_backbone.vision.class_embedding" in model.loaded
    assert "vision.class_embedding" not in model.loaded
    assert "lm.embeddings.weight" in model.loaded


def test_unsharded_export_canonicalization_is_idempotent(tmp_path: Path) -> None:
    # A current-layout export must round-trip untouched, not gain a second prefix.
    checkpoint = _write_export(tmp_path, {"vision_backbone.connector.w1.weight": torch.zeros(2, 2)})

    model = _FakeModel()
    load_checkpoint_weights(_info("olmo_core_unsharded"), str(checkpoint), model)

    assert model.loaded is not None
    assert list(model.loaded) == ["vision_backbone.connector.w1.weight"]


def test_unsharded_path_uses_the_olmo_core_helper() -> None:
    # Guards against someone re-hand-rolling the prefix logic here: the rename rules live
    # in OLMo-core and must not be duplicated in the eval harness.
    import inspect

    source = inspect.getsource(vlm_utils.load_checkpoint_weights)
    assert "canonicalize_vision_keys" in source
