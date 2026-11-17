"""Tests for the typed lazy ImageReference contract."""

from __future__ import annotations

import base64
import io
import json
import pickle
from dataclasses import dataclass

import pytest
from rich.console import Console

from olmo_eval.common.images import (
    CappedImageList,
    HfDatasetImage,
    HfDatasetImageList,
    ImageList,
    ImageReference,
    PathImage,
    resolve_image,
    resolve_images,
)
from olmo_eval.common.inspection import format_value, inspect_request
from olmo_eval.common.types import Instance, LMRequest, RequestType
from olmo_eval.evals.tasks.common.base import TaskConfig
from olmo_eval.evals.vision.benchmarks.cc_ocr import TsvImageReference
from olmo_eval.evals.vision.benchmarks.mmmu_pro import MmmuProTask
from olmo_eval.evals.vision.benchmarks.olmocr_bench import PdfPageImage
from olmo_eval.evals.vision.data.images import (
    capped_image_list,
    lazy_hf_image,
    lazy_hf_image_list,
    load_instance_image,
)
from olmo_eval.runners.asynq.progress import _metadata_changes, request_hash
from olmo_eval.runners.asynq.types import QueueItem


@dataclass(frozen=True)
class _StubDatasetInfo:
    dataset_name: str


class _StubHfDataset:
    """Module-scope dataset stand-in to exercise real pickle round-trips."""

    def __init__(
        self,
        rows: list[dict],
        *,
        dataset_name: str = "demo/vision",
        split: str = "validation",
    ) -> None:
        self._rows = rows
        self.info = _StubDatasetInfo(dataset_name=dataset_name)
        self.split = split

    def __getitem__(self, index: int) -> dict:
        return self._rows[index]


def _png_bytes(size: tuple[int, int] = (4, 3), color: str = "red") -> bytes:
    PIL_Image = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    PIL_Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


class TestImageReferenceContract:
    def test_path_image_is_lazy_picklable_and_json_serializable(self, tmp_path):
        PIL_Image = pytest.importorskip("PIL.Image")

        missing = PathImage(tmp_path / "does_not_exist.png")
        assert isinstance(missing, ImageReference)
        assert missing.to_json() == {"type": "path", "path": str(tmp_path / "does_not_exist.png")}

        img_path = tmp_path / "sample.png"
        PIL_Image.new("RGB", (6, 5)).save(img_path)
        ref = pickle.loads(pickle.dumps(PathImage(img_path)))
        with ref.resolve() as loaded:
            assert loaded.size == (6, 5)
        with resolve_image(ref) as loaded_via_helper:
            assert loaded_via_helper.size == (6, 5)

    def test_hf_dataset_image_decodes_bytes_and_paths_and_roundtrips_pickle(self, tmp_path):
        PIL_Image = pytest.importorskip("PIL.Image")

        on_disk = tmp_path / "cell.png"
        PIL_Image.new("RGB", (7, 4)).save(on_disk)
        ds = _StubHfDataset(
            rows=[
                {"image": {"bytes": _png_bytes((5, 2)), "path": None}},
                {"image": {"bytes": None, "path": str(on_disk)}},
            ]
        )

        ref0 = lazy_hf_image(ds, 0, "image")
        ref1 = lazy_hf_image(ds, 1, "image")
        assert isinstance(ref0, HfDatasetImage)
        assert ref0.to_json() == {
            "type": "hf_dataset_image",
            "index": 0,
            "column": "image",
            "dataset": "demo/vision",
            "split": "validation",
        }
        assert json.loads(json.dumps(ref0.to_json())) == ref0.to_json()

        restored0 = pickle.loads(pickle.dumps(ref0))
        restored1 = pickle.loads(pickle.dumps(ref1))
        assert restored0.resolve().size == (5, 2)
        with restored1.resolve() as img1:
            assert img1.size == (7, 4)

        inst = Instance(question="q", metadata={"image": ref0})
        assert load_instance_image(inst).size == (5, 2)

    def test_hf_dataset_image_list_and_capped_list_truncate_before_decode(self):
        pytest.importorskip("PIL.Image")

        # Put corrupt PNG bytes past max_images=2 to verify slicing runs before decoding.
        ds = _StubHfDataset(
            rows=[
                {
                    "image_list": [
                        {"bytes": _png_bytes((2, 2))},
                        {"bytes": _png_bytes((3, 2))},
                        {"bytes": b"not-a-valid-png"},
                    ]
                }
            ]
        )
        list_ref = lazy_hf_image_list(ds, 0, "image_list")
        assert isinstance(list_ref, HfDatasetImageList)
        capped = capped_image_list(list_ref, max_images=2)
        assert isinstance(capped, CappedImageList)

        restored = pickle.loads(pickle.dumps(capped))
        resolved = resolve_images((restored,))
        assert resolved is not None
        assert [img.size for img in resolved] == [(2, 2), (3, 2)]
        assert restored.to_json() == {
            "type": "capped_image_list",
            "max_images": 2,
            "images": {
                "type": "hf_dataset_image_list",
                "index": 0,
                "column": "image_list",
                "dataset": "demo/vision",
                "split": "validation",
            },
        }

    def test_image_list_validation_pickling_and_mmmu_pro_integration(self, tmp_path):
        PIL_Image = pytest.importorskip("PIL.Image")

        with pytest.raises(TypeError, match="sequence"):
            ImageList("not-a-sequence")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="max_images"):
            CappedImageList((), max_images=-1)

        img_path = tmp_path / "p.png"
        PIL_Image.new("RGB", (4, 4)).save(img_path)
        ds = _StubHfDataset(rows=[{"image": {"bytes": _png_bytes((3, 6))}}])
        image_list = ImageList((PathImage(img_path), lazy_hf_image(ds, 0, "image"), None))
        restored = pickle.loads(pickle.dumps(image_list))
        resolved = resolve_images((restored,))
        assert resolved is not None
        assert [img.size for img in resolved] == [(4, 4), (3, 6)]
        resolved[0].close()

        # Verify MmmuProTask._attach_images handles typed ImageList without eager decoding.
        task = MmmuProTask(TaskConfig(name="mmmu_pro"))
        inst = Instance(question="Q", metadata={"images": ImageList((lazy_hf_image(ds, 0),))})
        req = task.format_request(inst)
        assert req.images is not None
        assert isinstance(req.images[0], ImageList)
        assert (
            task.format_request(Instance(question="Q", metadata={"images": ImageList(())})).images
            is None
        )

    def test_tsv_and_pdf_references_pickle_and_serialize(self, tmp_path):
        pytest.importorskip("PIL.Image")

        encoded = base64.b64encode(_png_bytes((9, 4))).decode("ascii")
        tsv = tmp_path / "mini.tsv"
        tsv.write_text(f"image\tcategory\n{encoded}\tocr\n", encoding="utf-8")

        tsv_ref = pickle.loads(pickle.dumps(TsvImageReference(path=str(tsv), row=0)))
        assert isinstance(tsv_ref, ImageReference)
        assert tsv_ref.resolve().size == (9, 4)
        assert tsv_ref.to_json() == {"type": "tsv_image", "path": str(tsv), "row": 0}

        pdf_ref = pickle.loads(
            pickle.dumps(PdfPageImage(pdf_path="/tmp/doc.pdf", target_longest_image_dim=2048))
        )
        assert isinstance(pdf_ref, ImageReference)
        assert pdf_ref.to_json() == {
            "type": "pdf_page",
            "pdf_path": "/tmp/doc.pdf",
            "target_longest_image_dim": 2048,
        }

    def test_request_hash_progress_and_inspection_handle_image_references(self):
        # Dataset is intentionally non-JSON-serializable (unhashable Mock-like object);
        # hashing must use ImageReference.to_json() rather than dataclasses.asdict().
        class _OpaqueDataset:
            pass

        opaque = _OpaqueDataset()
        ref0 = HfDatasetImage(dataset=opaque, index=0, column="image")
        ref1 = HfDatasetImage(dataset=opaque, index=1, column="image")

        item0 = QueueItem(
            model_name="m",
            task_id="ai2d",
            instance_idx=0,
            instance=Instance(question="Q", gold_answer="A", metadata={"image": ref0}),
            request=LMRequest(
                request_type=RequestType.CHAT,
                messages=({"role": "user", "content": "Q"},),
                images=(ref0,),
            ),
        )
        item1 = QueueItem(
            model_name="m",
            task_id="ai2d",
            instance_idx=1,
            instance=Instance(question="Q", gold_answer="A", metadata={"image": ref1}),
            request=LMRequest(
                request_type=RequestType.CHAT,
                messages=({"role": "user", "content": "Q"},),
                images=(ref1,),
            ),
        )

        assert request_hash(item0) == request_hash(item0)
        assert request_hash(item0) != request_hash(item1)
        assert _metadata_changes(item0.instance.metadata, {"image": ref0, "extra": 1}) == {
            "extra": 1
        }

        formatted = format_value({"image": ref0})
        assert json.loads(formatted) == {
            "image": {"type": "hf_dataset_image", "index": 0, "column": "image"}
        }

        console = Console(file=io.StringIO(), force_terminal=False)
        inspect_request(item0.request, console=console, task_name="ai2d", native_id="0")
        rendered = console.file.getvalue()
        assert "hf_dataset_image" in rendered
