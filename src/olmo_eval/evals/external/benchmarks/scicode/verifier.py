"""SciCode per-sub-step test script construction and execution.

Inlines the upstream ``scicode/parse/parse.py::process_hdf5_to_tuple`` helper so
the sandbox does not need the ``scicode`` package installed. The sandbox only
needs ``numpy``, ``scipy``, ``sympy``, and ``h5py`` plus the numeric reference
file ``test_data.h5`` at ``h5py_file``.
"""

from __future__ import annotations

from typing import Any

_H5_HELPER = """
import h5py as _h5py
import numpy as _np
import scipy as _scipy

def _scicode_process_hdf5_list(group):
    return [group[key][()] for key in group.keys()]

def _scicode_process_hdf5_sparse_matrix(group):
    data = group["data"][()]
    shape = tuple(group["shape"][()])
    if "row" in group and "col" in group:
        row = group["row"][()]
        col = group["col"][()]
        return _scipy.sparse.coo_matrix((data, (row, col)), shape=shape)
    if "blocksize" in group:
        indices = group["indices"][()]
        indptr = group["indptr"][()]
        blocksize = tuple(group["blocksize"][()])
        return _scipy.sparse.bsr_matrix(
            (data, indices, indptr), shape=shape, blocksize=blocksize
        )
    indices = group["indices"][()]
    indptr = group["indptr"][()]
    return _scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)

def _scicode_process_hdf5_dict(group):
    out = {}
    for key, obj in group.items():
        if isinstance(obj, _h5py.Group):
            out[key] = _scicode_process_hdf5_sparse_matrix(obj["sparse_matrix"])
        elif isinstance(obj[()], bytes):
            out[key] = obj[()].decode("utf-8", errors="strict")
        else:
            try:
                out[float(key)] = obj[()]
            except ValueError:
                out[key] = obj[()]
    return out

def _scicode_process_hdf5_datagroup(group):
    for key in group.keys():
        if key == "list":
            return _scicode_process_hdf5_list(group[key])
        if key == "sparse_matrix":
            return _scicode_process_hdf5_sparse_matrix(group[key])
        return _scicode_process_hdf5_dict(group)

def process_hdf5_to_tuple(step_id, test_num, h5py_file):
    data_lst = []
    with _h5py.File(h5py_file, "r") as f:
        for test_id in range(test_num):
            group_path = f"{step_id}/test{test_id + 1}"
            node = f[group_path]
            if not isinstance(node, _h5py.Group):
                raise FileNotFoundError(f"Path {group_path} not found in the file.")
            keys = list(node.keys())
            if len(keys) == 1:
                sub = node[keys[0]]
                if isinstance(sub, _h5py.Dataset):
                    val = sub[()]
                    data_lst.append(val.decode("utf-8") if isinstance(val, bytes) else val)
                else:
                    data_lst.append(_scicode_process_hdf5_datagroup(sub))
            else:
                var_lst = []
                for key in node.keys():
                    sub = node[key]
                    if isinstance(sub, _h5py.Dataset):
                        val = sub[()]
                        var_lst.append(val.decode("utf-8") if isinstance(val, bytes) else val)
                    else:
                        var_lst.append(_scicode_process_hdf5_datagroup(sub))
                data_lst.append(tuple(var_lst))
    return data_lst
"""


def build_step_script(
    step: dict[str, Any],
    required_dependencies: str,
    full_code: str,
    hardcoded_prelude: str,
    h5py_file: str,
) -> str:
    """Build the Python test script for a single sub-step.

    The script imports required dependencies, pulls numeric targets from the
    reference h5py file, and executes the step's test assertions against
    ``full_code`` (the concatenation of all generated sub-step code).
    """
    step_id = step["step_number"]
    test_cases = list(step["test_cases"])
    lines: list[str] = [required_dependencies.strip(), _H5_HELPER.strip()]
    if hardcoded_prelude.strip():
        lines.append(hardcoded_prelude.strip())
    lines.append(full_code.strip())
    lines.append(f"targets = process_hdf5_to_tuple({step_id!r}, {len(test_cases)}, {h5py_file!r})")
    for idx, tc in enumerate(test_cases):
        lines.append(f"target = targets[{idx}]")
        lines.append(tc)
    return "\n\n".join(lines) + "\n"
