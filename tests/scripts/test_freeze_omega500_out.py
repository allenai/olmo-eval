"""Exercise manifest generation against local snapshots without network access."""

import ast
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/hillclimb/freeze_omega500_out.py"


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def generate(root, output):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(root), str(output)],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_manifest_selection_is_frozen_and_order_independent(tmp_path):
    families = ["geometry_rotation", "geometry_rotation", "number_theory_prime_mod", "other"]
    anchor = tmp_path / "omega-500/train-00000-of-00001.parquet"
    candidates = {
        "geometry_polygon_rotation": [f"geom{i}" for i in range(5)],
        "numbertheory_lte_qr": [f"prime{i}" for i in range(5)],
        "other": [f"other{i}" for i in range(5)],
    }
    outputs = []
    for reverse in [False, True]:
        ordered = list(reversed(families)) if reverse else families
        write_rows(anchor, [{"family": family} for family in ordered])
        for config, ids in candidates.items():
            ordered_ids = list(reversed(ids)) if reverse else ids
            # Deliberately omit answers: selection must depend only on IDs and families.
            write_rows(
                tmp_path / "omega-explorative" / config / "test_out-00000-of-00001.parquet",
                [{"id": item_id} for item_id in ordered_ids],
            )
        output = tmp_path / f"manifest-{reverse}.py"
        result = generate(tmp_path, output)
        assert result.returncode == 0, result.stderr
        outputs.append(output.read_text())
    assert outputs[0] == outputs[1]
    values = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(outputs[0]).body
        if isinstance(node, ast.Assign)
    }
    assert values["FAMILY_COUNTS"] == {
        "geometry_rotation": 2,
        "number_theory_prime_mod": 1,
        "other": 1,
    }
    assert values["IDS_BY_CONFIG"] == {
        "geometry_polygon_rotation": ("geom3", "geom4"),
        "numbertheory_lte_qr": ("prime4",),
        "other": ("other3",),
    }
    assert values["FAMILY_ALIASES"]["geometry_rotation"] == "geometry_polygon_rotation"
    assert values["FAMILY_ALIASES"]["number_theory_prime_mod"] == "numbertheory_lte_qr"


@pytest.mark.parametrize("ids", [["a"], ["a", "a", "b"]], ids=["insufficient", "duplicate"])
def test_invalid_candidate_pool_does_not_write_manifest(tmp_path, ids):
    write_rows(
        tmp_path / "omega-500/train-00000-of-00001.parquet",
        [{"family": "other"}, {"family": "other"}],
    )
    write_rows(
        tmp_path / "omega-explorative/other/test_out-00000-of-00001.parquet",
        [{"id": item_id} for item_id in ids],
    )
    output = tmp_path / "manifest.py"
    result = generate(tmp_path, output)
    assert result.returncode != 0
    assert "AssertionError" in result.stderr
    assert not output.exists()
