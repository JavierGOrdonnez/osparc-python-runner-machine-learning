#!/usr/bin/env python3
"""Layer 1: pure tests for the GPU regression validation logic.

No Docker, no GPU, no I/O against the runner. Runs anywhere Python runs.
Exercises validate_against_golden plus a structural check of the shipped
golden fixtures so a malformed fixture is caught before any container starts.
"""

import json
import sys
import unittest
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]
ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))

from validate_ml_regression import validate_against_golden  # noqa: E402


def _good_result():
    return {
        "predictions": [0, 1, 2],
        "probabilities": [
            [0.9997533, 0.0001233794, 0.0001233794],
            [0.0003352377, 0.999329567, 0.0003352377],
            [0.0009102219, 0.0009102219, 0.998179555],
        ],
        "actual_device": "gpu",
    }


def _golden():
    return {
        "predictions": [0, 1, 2],
        "probabilities": [
            [0.9997533, 0.0001233794, 0.0001233794],
            [0.0003352377, 0.999329567, 0.0003352377],
            [0.0009102219, 0.0009102219, 0.998179555],
        ],
        "probability_tolerance": 0.00001,
    }


class TestValidateAgainstGolden(unittest.TestCase):
    def test_matching_result_passes(self):
        validate_against_golden(_good_result(), _golden())

    def test_prediction_mismatch_fails(self):
        result = _good_result()
        result["predictions"] = [2, 1, 0]
        with self.assertRaises(AssertionError):
            validate_against_golden(result, _golden())

    def test_probability_out_of_tolerance_fails(self):
        result = _good_result()
        result["probabilities"][0][0] = 0.5
        with self.assertRaises(AssertionError):
            validate_against_golden(result, _golden())

    def test_probability_within_tolerance_passes(self):
        result = _good_result()
        result["probabilities"][0][0] = 0.9997533 + 0.000005
        validate_against_golden(result, _golden())

    def test_cpu_fallback_fails(self):
        result = _good_result()
        result["actual_device"] = "cpu"
        with self.assertRaises(AssertionError):
            validate_against_golden(result, _golden())


class TestShippedFixtures(unittest.TestCase):
    """The golden files that Layer 3 compares against must be well-formed."""

    def test_fixtures_are_structurally_valid(self):
        for framework in ("pytorch", "tensorflow"):
            golden_path = (
                ROOT / f"regression-{framework}" / "inputs" / "input_1" / "golden.json"
            )
            self.assertTrue(golden_path.is_file(), f"missing {golden_path}")
            golden = json.loads(golden_path.read_text())
            for key in ("predictions", "probabilities", "probability_tolerance"):
                self.assertIn(key, golden)
            self.assertIsInstance(golden["probability_tolerance"], (int, float))
            self.assertGreater(golden["probability_tolerance"], 0)
            # A result cloned from its own golden must validate (device aside).
            result = {
                "predictions": golden["predictions"],
                "probabilities": golden["probabilities"],
                "actual_device": "gpu",
            }
            validate_against_golden(result, golden)


if __name__ == "__main__":
    unittest.main(verbosity=2)
