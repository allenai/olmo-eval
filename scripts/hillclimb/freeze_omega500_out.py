"""Reproduce the frozen OMEGA out manifest from pinned local parquet snapshots."""

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "snapshot_root",
    type=Path,
    help="Directory with omega-500/ and omega-explorative/ snapshots at the manifest revisions",
)
parser.add_argument(
    "output", type=Path, help="Output Python manifest; format with ruff after generation"
)
args = parser.parse_args()
root = args.snapshot_root
aliases = {
    "geometry_rotation": "geometry_polygon_rotation",
    "logic_puzzles_blocked_grid": "logic_gridworld_blocked",
    "logic_puzzles_grid_knight": "logic_gridworld_knight_move",
    "logic_puzzles_grid_rook": "logic_gridworld_rookmove",
    "logic_puzzles_zebralogic": "logic_zebralogic",
    "number_theory_digit_sum": "numbertheory_qr_sum",
    "number_theory_prime_mod": "numbertheory_lte_qr",
    "number_theory_triple_count": "numbertheory_ordered_lte",
}
counts = Counter(
    r["family"] for r in pq.read_table(root / "omega-500/train-00000-of-00001.parquet").to_pylist()
)
text = '''"""Frozen OMEGA test_out dev IDs, stratified to match omega-500 families.

Revisions: omega-500 113a7eb896b8c1f7d781eb5f00713972074bae42;
omega-explorative b04ae8d4757a2229e8ed65ac6a923e502d0bbb95.
Within each mapped family, select the required count by ascending SHA256
of ``olmo35-omega-out-v1:<id>``. No answers or model results enter selection.
The alias map reconciles the datasets' family naming; the number-theory
families map by task definition (digit constraints, prime powers, tuple counts).
"""

'''
text += "FAMILY_ALIASES = " + repr(aliases) + "\n\n"
text += "FAMILY_COUNTS = " + repr(dict(sorted(counts.items()))) + "\n\n"
text += "IDS_BY_CONFIG = {\n"
for family, n in sorted(counts.items()):
    config = aliases.get(family, family)
    rows = pq.read_table(
        root / "omega-explorative" / config / "test_out-00000-of-00001.parquet"
    ).to_pylist()
    ids = sorted(r["id"] for r in rows)
    assert len(ids) == len(set(ids)) and len(ids) >= n
    chosen = sorted(
        sorted(
            ids, key=lambda x: hashlib.sha256(("olmo35-omega-out-v1:" + x).encode()).hexdigest()
        )[:n]
    )
    text += f'    "{config}": (\n' + "".join(f'        "{i}",\n' for i in chosen) + "    ),\n"
text += "}\n"
args.output.write_text(text)
print("Frozen", sum(counts.values()), "IDs across", len(counts), "families")
