"""Round-trip tests for eval-arg / provider-kwarg serialization.

The Beaker launcher serializes parsed args back into ``-a key=value`` strings
that the inner ``run-external`` CLI re-parses. ``json.dumps`` on the way out and
``json.loads`` (via ``_coerce_value``) on the way in must agree on every shape
we care about: strings, bools, ints, floats, lists, dicts, None.
"""

from __future__ import annotations

import json
import unittest
from typing import Any

from parameterized import parameterized

from olmo_eval.cli import utils as cli_utils


class TestArgRoundTrip(unittest.TestCase):
    @parameterized.expand(
        [
            ("string", "hello"),
            ("string_with_spaces", "hello world"),
            ("bool_true", True),
            ("bool_false", False),
            ("int", 123),
            ("float", 3.14),
            ("list_of_strings", ["2", "11"]),
            ("list_of_ints", [1, 2, 3]),
            ("dict", {"a": 1, "b": "two"}),
            ("none", None),
        ]
    )
    def test_round_trip(self, _name: str, value: Any) -> None:
        encoded = json.dumps(value)
        parsed = cli_utils.parse_key_value_args((f"k={encoded}",), coerce_types=True)
        self.assertEqual(parsed["k"], value)


if __name__ == "__main__":
    unittest.main()
