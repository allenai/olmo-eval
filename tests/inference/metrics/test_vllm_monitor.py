"""Tests for VLLMMetricsMonitor."""

from olmo_eval.inference.metrics.core.vllm_monitor import (
    VLLMMetricsSnapshot,
    parse_prometheus_metrics,
)


class TestParsePrometheusMetrics:
    """Tests for parse_prometheus_metrics function."""

    def test_parse_simple_gauge(self) -> None:
        """Parse a simple gauge metric."""
        text = "vllm:num_requests_running 5"
        result = parse_prometheus_metrics(text)
        assert result == {"vllm:num_requests_running": {"labels": {}, "value": 5.0}}

    def test_parse_gauge_with_labels(self) -> None:
        """Parse a gauge metric with labels."""
        text = 'vllm:request_success_total{finished_reason="stop"} 42'
        result = parse_prometheus_metrics(text)
        assert result == {
            "vllm:request_success_total": {
                "labels": {"finished_reason": "stop"},
                "value": 42.0,
            }
        }

    def test_parse_float_value(self) -> None:
        """Parse a floating point value."""
        text = "vllm:kv_cache_usage_perc 0.456"
        result = parse_prometheus_metrics(text)
        assert result == {"vllm:kv_cache_usage_perc": {"labels": {}, "value": 0.456}}

    def test_parse_scientific_notation(self) -> None:
        """Parse scientific notation values."""
        text = "vllm:tokens_total 1.5e6"
        result = parse_prometheus_metrics(text)
        assert result == {"vllm:tokens_total": {"labels": {}, "value": 1.5e6}}

    def test_skip_comments(self) -> None:
        """Skip comment lines."""
        text = """# HELP vllm:num_requests_running Number of requests running
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 5"""
        result = parse_prometheus_metrics(text)
        assert result == {"vllm:num_requests_running": {"labels": {}, "value": 5.0}}

    def test_skip_empty_lines(self) -> None:
        """Skip empty lines."""
        text = """vllm:metric_a 1

vllm:metric_b 2
"""
        result = parse_prometheus_metrics(text)
        assert result == {
            "vllm:metric_a": {"labels": {}, "value": 1.0},
            "vllm:metric_b": {"labels": {}, "value": 2.0},
        }

    def test_parse_multiple_metrics(self) -> None:
        """Parse multiple metrics."""
        text = """vllm:num_requests_running 5
vllm:num_requests_waiting 3
vllm:kv_cache_usage_perc 0.75"""
        result = parse_prometheus_metrics(text)
        assert result == {
            "vllm:num_requests_running": {"labels": {}, "value": 5.0},
            "vllm:num_requests_waiting": {"labels": {}, "value": 3.0},
            "vllm:kv_cache_usage_perc": {"labels": {}, "value": 0.75},
        }

    def test_parse_histogram_bucket(self) -> None:
        """Parse histogram bucket metrics with labels."""
        text = 'vllm:time_to_first_token_seconds_bucket{le="0.1"} 10'
        result = parse_prometheus_metrics(text)
        assert result == {
            "vllm:time_to_first_token_seconds_bucket": {"labels": {"le": "0.1"}, "value": 10.0}
        }

    def test_parse_negative_value(self) -> None:
        """Parse negative values (rare but valid)."""
        text = "some_metric -5.5"
        result = parse_prometheus_metrics(text)
        assert result == {"some_metric": {"labels": {}, "value": -5.5}}


class TestVLLMMetricsSnapshot:
    """Tests for VLLMMetricsSnapshot dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        snapshot = VLLMMetricsSnapshot(
            timestamp="2026-02-25T01:00:00+00:00",
            fetch_latency_ms=12.5,
            parsed_metrics={"vllm:num_requests_running": {"labels": {}, "value": 5.0}},
        )
        result = snapshot.to_dict()
        assert result == {
            "timestamp": "2026-02-25T01:00:00+00:00",
            "fetch_latency_ms": 12.5,
            "error": None,
            "metrics": {"vllm:num_requests_running": {"labels": {}, "value": 5.0}},
        }

    def test_to_dict_with_error(self) -> None:
        """Test conversion to dictionary with error."""
        snapshot = VLLMMetricsSnapshot(
            timestamp="2026-02-25T01:00:00+00:00",
            fetch_latency_ms=5000.0,
            error="Connection refused",
        )
        result = snapshot.to_dict()
        assert result["error"] == "Connection refused"
        assert result["metrics"] == {}
