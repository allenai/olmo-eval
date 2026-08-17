"""Tests for non-TTY progress reporting."""

from __future__ import annotations

import logging
from unittest.mock import Mock

from olmo_eval.common.progress import ProgressLogger


def test_incomplete_progress_never_rounds_to_one_hundred_percent() -> None:
    logger = Mock(spec=logging.Logger)
    progress = ProgressLogger(total=1000, logger=logger, log_interval=0)

    progress.update(999)
    assert "99.90%" in logger.info.call_args.args[0]
    assert "100%" not in logger.info.call_args.args[0]

    progress.close()
    assert "99.90%" in logger.info.call_args.args[0]
    assert "100%" not in logger.info.call_args.args[0]


def test_complete_progress_reports_one_hundred_percent() -> None:
    logger = Mock(spec=logging.Logger)
    progress = ProgressLogger(total=2, logger=logger, log_interval=0)

    progress.update(2)
    progress.close()

    assert "100%" in logger.info.call_args.args[0]
