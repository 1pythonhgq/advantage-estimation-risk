"""Contract checks with project dependencies; no model loading or vLLM inference."""

import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from diagnostic_reference_utils import (
    completed_response, reference_seed, score_response, summarize_reference,
)
from compute_token_reference_advantages_vllm import compute_record, load_jsonl_records


class CharacterTokenizer:
    eos_token_id = 0

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids if i != self.eos_token_id)


class ReferenceTests(unittest.TestCase):
    def test_jsonl_preserves_unicode_separators(self):
        expected = [dict(row_index=0, question="Clive.\u2028The box.\u2029How many?\u0085"),
                    dict(row_index=1, question="Second question\nwith escaped newline")]
        for newline in ("\n", "\r\n"):
            text = newline.join(json.dumps(r, ensure_ascii=False) for r in expected) + newline
            # The old loader split a valid JSON string and raised this error.
            with self.assertRaises(json.JSONDecodeError):
                [json.loads(line) for line in text.splitlines() if line.strip()]
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "trajectories.jsonl"
                path.write_bytes(text.encode("utf-8"))
                self.assertEqual(load_jsonl_records(path), expected)
                self.assertEqual(load_jsonl_records(path, limit=1), expected[:1])

    def test_jsonl_reports_actual_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.jsonl"
            path.write_text('{"row_index": 0}\n{"question": "cut off', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r":2, column"):
                load_jsonl_records(path)

    def test_values_terminal_reward_and_telescoping(self):
        result = summarize_reference([[0, 0, 1, 1], [1, 1, 1, 0]], 1)
        self.assertEqual(result["values"], [0.5, 0.75, 0])
        self.assertEqual(result["token_advantages"], [0.25, 0.25])
        self.assertAlmostEqual(result["telescoping_residual"], 0)
        self.assertAlmostEqual(result["value_standard_errors"][0], math.sqrt(1/12))

    def test_one_token_and_negative_penalty(self):
        result = summarize_reference([[1, 1, 1, 1]], -2)
        self.assertEqual(result["token_advantages"], [-3])
        self.assertEqual(result["values"][-1], 0)

    def test_invalid_samples(self):
        for rewards in ([], [[1]], [[1, 0], [1, 0, 1]], [[float("nan"), 0]]):
            with self.assertRaises(ValueError):
                summarize_reference(rewards, 1)

    def test_request_seed_namespaces(self):
        seeds = [reference_seed(42, g, r, q, t)
                 for g in (1, 2) for r in range(3) for q in range(3) for t in range(5)]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertGreater(min(seeds), 4_000_000)
        self.assertLess(reference_seed(2**32-1, 2, 1023, 2**20-1, 65535), 2**63)
        with self.assertRaises(ValueError):
            reference_seed(42, 1, 0, 0, 65536)

    def test_reward_modes_and_length_cutoff(self):
        self.assertEqual(score_response("work #### 1,000", "1000", "stop"), 1)
        self.assertEqual(score_response("work #### 1000", "1000", "length"), 0)
        self.assertEqual(score_response("#### 1 #### 1", "1", "stop"), -2)
        self.assertEqual(score_response("#### 1 #### 1", "1", "length"), 0)
        self.assertEqual(score_response("answer is 1", "1", "stop"), 0)
        self.assertEqual(score_response("#### 2", "1", "stop"), 0)
        self.assertEqual(score_response("#### 1.0", "1", "stop"), 0)
        self.assertEqual(score_response("#### 1,000", "1,000", "stop"), 0)
        self.assertEqual(score_response("#### 1 #### 1", "1", "stop", "trajectory"), -2)
        with self.assertRaises(ValueError):
            score_response("#### 1", "1", "stop", "binary")

    def test_stop_crossing_prefix_boundary(self):
        tokenizer = CharacterTokenizer()
        fixed = "work #### 2\n\n"
        continuation = "\nProblem: trailing output"
        text, finish = completed_response(tokenizer, list(map(ord, "Question:")),
                                          list(map(ord, fixed + continuation)),
                                          "\n\n\nProblem:", "length")
        self.assertEqual(text, "work #### 2")
        self.assertEqual(finish, "stop")
        self.assertEqual(score_response(text, "2", finish), 1)

    def test_all_prefixes_budgets_and_independent_groups(self):
        class FakeLLM:
            def __init__(self):
                self.calls = []

            def generate(self, *, prompt_token_ids, sampling_params, use_tqdm):
                self.calls.append((prompt_token_ids, sampling_params))
                completions = [SimpleNamespace(token_ids=list(map(ord, "#### 2")),
                                                text="#### 2", finish_reason="stop")
                               for _ in range(sampling_params.n)]
                return [SimpleNamespace(prompt_token_ids=prompt_token_ids[0], outputs=completions)]

        engine = FakeLLM()
        record = dict(row_index=7, source_index=70, selected_index=0,
                      gold_answer="2", prompt_token_ids=[ord("Q")],
                      selected_rollout=dict(response_token_ids=[ord("a"), ord("b")],
                                            response_text="ab", finish_reason="length", reward=0))
        metadata = dict(seed=42, temperature=0.6, top_p=0.9,
                        max_new_tokens=20, stop_text="\n\n\nProblem:")
        audit = io.StringIO()
        result = compute_record(record, engine, SimpleNamespace, CharacterTokenizer(), metadata,
                                SimpleNamespace(rounds=1, mc_samples=4, reward_mode="project"), audit)
        self.assertEqual([call[0] for call in engine.calls], [[[81]], [[81, 97]], [[81]], [[81, 97]]])
        self.assertEqual([call[1].max_tokens for call in engine.calls], [20, 19, 20, 19])
        self.assertEqual(len({call[1].seed for call in engine.calls}), 4)
        for group in result["rounds"][0]["groups"]:
            self.assertEqual(group["token_advantages"], [0, -1])
            self.assertEqual(group["values"], [1, 1, 0])
        rows = [json.loads(line) for line in audit.getvalue().splitlines()]
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(len(row["completions"]) == 4 for row in rows))


if __name__ == "__main__":
    unittest.main()
