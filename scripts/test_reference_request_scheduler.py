"""CPU-only batching tests using an out-of-order fake vLLM 0.4 engine."""

import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from compute_token_reference_advantages_vllm import compute_records
from reference_request_scheduler import run_requests


def make_output(request_id, prompt, params, finished=True):
    # Variable rewards make misrouting across prefixes/trajectories observable.
    answer = "1" if params.seed % 3 == 0 else "2"
    text = "#### " + answer
    return SimpleNamespace(
        request_id=request_id, finished=finished, prompt_token_ids=prompt,
        outputs=[SimpleNamespace(token_ids=list(map(ord, text)), text=text,
                                 finish_reason="stop") for _ in range(params.n)],
    )


class FakeEngine:
    def __init__(self):
        self.pending = {}
        self.calls = []
        self.peak_pending = 0
        self.aborted = []
        self.steps = 0

    def add_request(self, request_id, prompt, params, prompt_token_ids):
        assert prompt is None
        self.pending[request_id] = (prompt_token_ids, params)
        self.calls.append((list(prompt_token_ids), params))
        self.peak_pending = max(self.peak_pending, len(self.pending))

    def has_unfinished_requests(self):
        return bool(self.pending)

    def step(self):
        self.steps += 1
        # Emit partial outputs first, then finish most recent request first.
        if self.steps == 1:
            return [make_output(k, *v, finished=False) for k, v in self.pending.items()]
        request_id = next(reversed(self.pending))
        return [make_output(request_id, *self.pending.pop(request_id))]

    def abort_request(self, request_id):
        self.aborted.append(request_id)
        self.pending.pop(request_id)


class FakeLLM:
    def __init__(self):
        self.llm_engine = FakeEngine()

    def generate(self, *, prompt_token_ids, sampling_params, use_tqdm):
        self.llm_engine.calls.append((prompt_token_ids[0], sampling_params))
        return [make_output("serial", prompt_token_ids[0], sampling_params)]


class Tokenizer:
    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))


def simple_score(text, gold, finish, mode):
    return float(finish == "stop" and text.split("####")[-1].strip() == gold)


class SchedulerTests(unittest.TestCase):
    def records(self):
        return [dict(row_index=row, source_index=row * 10, selected_index=0,
                     gold_answer=gold, prompt_token_ids=list(map(ord, prompt)),
                     selected_rollout=dict(response_token_ids=list(map(ord, response)),
                                           response_text=response, finish_reason="length", reward=0))
                for row, gold, prompt, response in [(7, "1", "Q", "ab"), (3, "2", "Z", "xyz")]]

    @patch("compute_token_reference_advantages_vllm.score_response", side_effect=simple_score)
    def test_serial_parallel_equivalence_and_request_parameters(self, scorer):
        records = self.records()
        metadata = dict(seed=42, temperature=0.6, top_p=0.9,
                        max_new_tokens=20, stop_text="\n\n\nProblem:")
        results = []
        audits = []
        engines = []
        for backend in ("serial", "engine"):
            llm = FakeLLM()
            audit = io.StringIO()
            args = SimpleNamespace(rounds=2, mc_samples=4, reward_mode="project",
                                   backend=backend, max_pending_requests=3)
            outputs = list(compute_records(records, llm, SimpleNamespace, Tokenizer(),
                                           metadata, args, audit))
            results.append(sorted(outputs, key=lambda r: r["row_index"]))
            audits.append(sorted((json.loads(line) for line in audit.getvalue().split("\n") if line),
                                 key=lambda r: (r["row_index"], r["round_index"], r["group"], r["prefix_length"])))
            engines.append(llm.llm_engine)
        self.assertEqual(results[0], results[1])
        self.assertEqual(audits[0], audits[1])
        self.assertEqual(len(audits[1]), (2 + 3) * 2 * 2)
        self.assertEqual(len({r["seed"] for r in audits[1]}), len(audits[1]))
        self.assertEqual(engines[1].peak_pending, 3)
        self.assertFalse(engines[1].pending)
        self.assertGreater(engines[1].steps, 3)  # Refilled after initial queue.
        self.assertEqual([c[0][0] for c in engines[1].calls[:2]], [ord("Q"), ord("Z")])
        for prompt, params in engines[1].calls:
            self.assertEqual(params.max_tokens, 20 - (len(prompt) - 1))
            self.assertEqual(params.n, 4)
            self.assertEqual(params.temperature, 0.6)
            self.assertEqual(params.top_p, 0.9)

    def jobs(self):
        for i in range(5):
            yield dict(prompt_token_ids=[i + 1], params=SimpleNamespace(seed=i, n=2))

    def test_closing_consumer_aborts_unfinished_requests(self):
        llm = FakeLLM()
        results = run_requests(llm, self.jobs(), 3)
        next(results)
        results.close()
        self.assertEqual(len(llm.llm_engine.aborted), 2)
        self.assertFalse(llm.llm_engine.pending)

    def test_reject_foreign_outputs_and_cleanup(self):
        llm = FakeLLM()
        llm.llm_engine.step = lambda: [SimpleNamespace(request_id="foreign", finished=True)]
        with self.assertRaisesRegex(ValueError, "Unknown"):
            list(run_requests(llm, self.jobs(), 3))
        self.assertFalse(llm.llm_engine.pending)

    def test_reject_busy_engine(self):
        llm = FakeLLM()
        llm.llm_engine.pending["existing"] = ([1], SimpleNamespace())
        with self.assertRaisesRegex(ValueError, "idle"):
            list(run_requests(llm, self.jobs(), 3))
        self.assertIn("existing", llm.llm_engine.pending)


if __name__ == "__main__":
    unittest.main()
