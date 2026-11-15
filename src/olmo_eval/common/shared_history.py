"""Retain artifacts that an opted-in shared supervisor would otherwise remove."""

from __future__ import annotations

import json
import os
import shutil
import stat
import time
import uuid
from pathlib import Path

from olmo_eval.common.constants.infrastructure import SHARED_RUN_ENV

RETAIN_SHARED_HISTORY_ENV = "OLMO_EVAL_RETAIN_SHARED_HISTORY"


def retained_shared_root(path: Path, *, root: Path | None = None) -> Path | None:
    """Match only paths in a session explicitly opted into retention.

    Children need both the supervisor environment and its saved policy. Ordinary local
    output cleanup and standalone progress locking keep their existing behavior.
    """
    required = root is None
    if root is None:
        directory = os.environ.get(SHARED_RUN_ENV)
        if os.environ.get(RETAIN_SHARED_HISTORY_ENV) != "1" or not directory:
            return None
        root = Path(directory)
    if root.is_symlink():
        raise RuntimeError("Retained shared root must not be a symlink")
    absolute, base = path.absolute(), root.absolute()
    if not absolute.is_relative_to(base):
        return None
    try:
        session = json.loads((root / "session.json").read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError("Retained shared session is missing or unreadable") from error
    if not isinstance(session, dict):
        raise RuntimeError("Retained shared session must be an object")
    if session.get("retain_shared_history") is not True:
        if required:
            raise RuntimeError("Retained shared session policy changed")
        return None
    if session.get("uuid") != root.name:
        raise RuntimeError("Retained shared session identity differs from its directory")
    if not path.resolve().is_relative_to(root.resolve()):
        raise RuntimeError("Retained shared artifact escapes its session")
    relative = absolute.relative_to(base)
    if not relative.parts or relative.parts[0] == "history":
        raise RuntimeError("Cannot retire a shared root or its history")
    return root


def archive_shared_path(
    path: Path, *, reason: str, root: Path | None = None, snapshot: bool = False
) -> bool:
    """Move a would-delete artifact to history; return False for ordinary cleanup.

    A compaction may snapshot its old inventory before atomically replacing it. This
    happens only when retiring segments, not on periodic progress commits. Missing
    artifacts need no action. No history is pruned by this option.
    """
    shared = retained_shared_root(path, root=root)
    if shared is None:
        return False
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return True
    if stat.S_ISLNK(mode):
        raise RuntimeError("Retained shared artifact must not be a symlink")
    relative = path.absolute().relative_to(shared.absolute())
    event = shared / "history" / uuid.uuid4().hex
    destination = event / relative
    destination.parent.mkdir(mode=0o700, parents=True)
    if snapshot:
        if not stat.S_ISREG(mode):
            raise RuntimeError("Only shared inventory files can be snapshotted")
        shutil.copy2(path, destination)
        with destination.open("rb") as stream:
            os.fsync(stream.fileno())
    else:
        os.replace(path, destination)
    with (event / "event.json").open("w") as stream:
        json.dump(
            {"source": str(relative), "reason": reason, "snapshot": snapshot, "time": time.time()},
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    # Persist every newly created directory link, not only the leaf containing the
    # moved inode, before a shared cleanup operation can continue.
    parents = {path.parent}
    parent = destination.parent
    while True:
        parents.add(parent)
        if parent == shared:
            break
        parent = parent.parent
    for parent in parents:
        fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return True
