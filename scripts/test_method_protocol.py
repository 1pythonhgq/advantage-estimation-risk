"""Stop-string regression tests; no numpy, torch, transformers or GPU required."""
import copy
import importlib.util
import json
import unittest

from project_method_adapter import load_project_protocol, validate_sampling_metadata


class SamplingMetadataTests(unittest.TestCase):
    def setUp(self):
        self.metadata = dict(temperature=0.6, top_p=0.9, max_new_tokens=1024,
                             stop_text="\n\n\nProblem:")
        # Exact representation after evaluating the project's Jsonnet stop field:
        # quotes are retained for guidance; newlines are literal characters.
        self.protocol = dict(sampling=dict(temperature=0.6, top_p=0.9, max_tokens=1024,
                                           stop='"\n\n\nProblem:"'))

    def test_literal_newlines_from_jsonnet(self):
        with self.assertRaises(json.JSONDecodeError):
            json.loads(self.protocol["sampling"]["stop"])
        before = copy.deepcopy((self.metadata, self.protocol))
        validate_sampling_metadata(self.metadata, self.protocol)
        self.assertEqual((self.metadata, self.protocol), before)

    def test_json_escaped_newlines_also_match(self):
        self.protocol["sampling"]["stop"] = json.dumps(self.metadata["stop_text"])
        validate_sampling_metadata(self.metadata, self.protocol)

    def test_different_newline_count_still_fails(self):
        self.metadata["stop_text"] = "\n\nProblem:"
        with self.assertRaisesRegex(ValueError, "stop text differs"):
            validate_sampling_metadata(self.metadata, self.protocol)

    def test_sampling_mismatch_still_fails(self):
        self.metadata["temperature"] = 1.0
        with self.assertRaisesRegex(ValueError, "temperature differs"):
            validate_sampling_metadata(self.metadata, self.protocol)

    def test_malformed_quoted_value_still_fails(self):
        self.protocol["sampling"]["stop"] = '"\n\n\nProblem:'
        with self.assertRaises(json.JSONDecodeError):
            validate_sampling_metadata(self.metadata, self.protocol)

    @unittest.skipUnless(importlib.util.find_spec("_jsonnet"), "Project Jsonnet dependency unavailable")
    def test_checked_in_project_configuration(self):
        validate_sampling_metadata(self.metadata, load_project_protocol())


if __name__ == "__main__":
    unittest.main()
