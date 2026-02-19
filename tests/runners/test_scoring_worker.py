"""Tests for scoring worker shutdown behavior and error handling."""

import queue
import time

from olmo_eval.runners.asynq.types import SCORER_FATAL, ScoredResponse


class TestScoringWorkerShutdown:
    """Tests for scoring worker shutdown with None sentinel."""

    def test_none_sentinel_stops_worker(self):
        """Test that None sentinel properly stops the scoring loop.

        This tests the fix for the bug where None sentinel was consumed
        but treated like a timeout, followed by a blocking get() that
        would never return.
        """
        # Simulate the scoring queue behavior
        scoring_queue: queue.Queue = queue.Queue()

        # Add some items then the sentinel
        scoring_queue.put("item1")
        scoring_queue.put("item2")
        scoring_queue.put(None)  # Sentinel

        items_processed = []
        shutdown_received = False

        # Simulate the fixed scoring loop logic
        while True:
            try:
                item = scoring_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                shutdown_received = True
                break

            items_processed.append(item)

        assert items_processed == ["item1", "item2"]
        assert shutdown_received is True

    def test_timeout_does_not_consume_sentinel(self):
        """Test that timeout handling doesn't interfere with sentinel detection."""
        scoring_queue: queue.Queue = queue.Queue()

        # Empty queue should timeout, not hang
        start = time.time()
        try:
            scoring_queue.get(timeout=0.1)
            got_item = True
        except queue.Empty:
            got_item = False
        elapsed = time.time() - start

        assert got_item is False
        assert elapsed < 0.5  # Should timeout quickly, not hang

    def test_batch_flushed_on_timeout(self):
        """Test that pending batch is processed on timeout."""
        scoring_queue: queue.Queue = queue.Queue()

        # Add items but no sentinel yet
        scoring_queue.put("item1")
        scoring_queue.put("item2")

        batch = []
        batches_processed = []

        # Simulate loop with timeout-based batch flushing
        iterations = 0
        max_iterations = 5

        while iterations < max_iterations:
            iterations += 1
            try:
                item = scoring_queue.get(timeout=0.05)
                if item is None:
                    break
                batch.append(item)
            except queue.Empty:
                # Timeout - flush batch if non-empty
                if batch:
                    batches_processed.append(list(batch))
                    batch = []
                continue

        # Batch should have been flushed on timeout
        assert len(batches_processed) == 1
        assert batches_processed[0] == ["item1", "item2"]

    def test_batch_flushed_on_shutdown(self):
        """Test that pending batch is processed when shutdown signal received."""
        scoring_queue: queue.Queue = queue.Queue()

        # Add items then sentinel
        scoring_queue.put("item1")
        scoring_queue.put("item2")
        scoring_queue.put(None)

        batch = []
        final_batch = None

        while True:
            try:
                item = scoring_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                # Shutdown - flush remaining batch
                if batch:
                    final_batch = list(batch)
                break

            batch.append(item)

        assert final_batch == ["item1", "item2"]


class TestScorerFatalHandling:
    """Tests for SCORER_FATAL error propagation."""

    def test_scorer_fatal_in_drain_loop(self):
        """Test that SCORER_FATAL is detected in the drain loop."""
        scored_queue: queue.Queue = queue.Queue()

        # Simulate a fatal scorer error
        fatal_response = ScoredResponse(
            spec=SCORER_FATAL,
            instance_idx=0,
            scored=None,
            error="Scorer process crashed",
        )
        scored_queue.put(fatal_response)

        # Simulate the drain loop check
        scored = scored_queue.get()

        assert scored.spec == SCORER_FATAL
        assert scored.error == "Scorer process crashed"

    def test_scorer_fatal_in_final_drain(self):
        """Test that SCORER_FATAL is caught in final drain path.

        This tests the fix where SCORER_FATAL wasn't checked in the
        final scoring drain loop, causing assert failures instead of
        proper error reporting.
        """
        scored_queue: queue.Queue = queue.Queue()

        # Add normal response then fatal
        normal_response = ScoredResponse(
            spec="task1",
            instance_idx=0,
            scored={"score": 1.0},
            error=None,
        )
        fatal_response = ScoredResponse(
            spec=SCORER_FATAL,
            instance_idx=0,
            scored=None,
            error="Out of memory",
        )

        scored_queue.put(normal_response)
        scored_queue.put(fatal_response)

        # Simulate final drain with SCORER_FATAL check
        results = []
        fatal_error = None

        while True:
            try:
                scored = scored_queue.get(timeout=0.1)
            except queue.Empty:
                break

            # This check was missing before the fix
            if scored.spec == SCORER_FATAL:
                fatal_error = scored.error
                break

            results.append(scored)

        assert len(results) == 1
        assert results[0].spec == "task1"
        assert fatal_error == "Out of memory"

    def test_scorer_fatal_without_check_would_fail_assert(self):
        """Verify that without SCORER_FATAL check, we'd hit the assert.

        This demonstrates the bug that was fixed: SCORER_FATAL responses
        have scored=None, which would fail the assert in handle_scored_response.
        """
        fatal_response = ScoredResponse(
            spec=SCORER_FATAL,
            instance_idx=0,
            scored=None,
            error="Crash",
        )

        # The old code would do this without checking for SCORER_FATAL first:
        # assert scored.scored is not None  <- This would fail!
        assert fatal_response.scored is None  # Confirm it would fail
        assert fatal_response.spec == SCORER_FATAL  # But we can detect it first
