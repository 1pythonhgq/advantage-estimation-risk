"""CPU regression tests execute actual project method bodies, without ML imports."""
import __future__
import ast
import copy
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest

import numpy as np

from compute_method_advantages_vllm import build_jobs, compute_records, validate_partitions
from diagnostic_reference_utils import reference_seed
from project_method_adapter import ProjectMethods, bind_methods, estimator_seed
from test_reference_request_scheduler import FakeLLM

ROOT = Path(__file__).resolve().parents[1] / "src/treetune"


def project_class(path, class_name, names):
    """Compile only selected original methods, avoiding torch/deepspeed imports."""
    source = ROOT / path
    parsed = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(body) == len(names)
    namespace = dict(np=np, json=json, defaultdict=defaultdict)
    for method in body:
        method.decorator_list = []
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return type(class_name, (), {name: namespace[name] for name in names})


class CharacterTokenizer:
    eos_token_id = 0
    bos_token_id = -1

    def __call__(self, text, **kwargs):
        return dict(input_ids=list(map(ord, text)), offset_mapping=[(i, i+1) for i in range(len(text))])

    def encode(self, text):
        # Model the default BOS of a text inference request.
        return [self.bos_token_id] + list(map(ord, text))

    def tokenize(self, text):
        return list(text)

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids if i >= 1)


def make_methods(tokenizer=None, cap=25):
    obj = object.__new__(ProjectMethods)
    obj.tokenizer = tokenizer or CharacterTokenizer()
    obj.protocol = dict(delimiter="", vine_max_steps=cap, spo_interval=5, model_context_size=200,
                        answer_extractor=dict(solution_prefix="\nSolution:", node_key_name="full_text"))
    task_cls = project_class("tasks/gsm8k.py", "GSM8K", ["split_solution_into_intermediate_steps"])
    obj.task = bind_methods(task_cls, SimpleNamespace(use_original_format=True, intermediate_step_delimiter="\n",
                            answer_prefix="\n#### "), ["split_solution_into_intermediate_steps"])
    vine_names = ["_create_step_query", "_get_value_estimation_queries", "_compute_step_advantages",
                  "_compute_token_advantages", "_compute_mc_value"]
    obj.vine_cls = project_class("episode_generators/vineppo_episode_generator.py", "VinePPOEpisodeGenerator", vine_names)
    class Reward:
        def __call__(self, query, answer, data):
            count = answer.count("####")
            if count > 1:
                return -2, False
            return float(count == 1 and answer.split("####")[-1].strip() == data["answer"]), False

        def get_unfinished_response_penalty(self):
            return 0.0
    obj.vine = bind_methods(obj.vine_cls, SimpleNamespace(tokenizer=obj.tokenizer, reasoning_step_delimiter="",
                            max_step_for_value_estimation=cap, reward_function=Reward()), vine_names)
    spo_names = ["_get_response_probs_and_cutpoints", "_create_cutpoint_query", "_compute_token_advantages", "_compute_mc_value"]
    spo_cls = project_class("episode_generators/math_episode_generator_with_mc_advantages.py",
                            "MathEpisodeGeneratorWithMCAdvantages", spo_names)
    obj.spo = bind_methods(spo_cls, SimpleNamespace(tokenizer=obj.tokenizer, cutpoint_interval=5,
                           reward_function=Reward()), spo_names)
    group_cls = project_class("episode_generators/math_episode_generator_with_group_advantages.py",
                               "MathEpisodeGeneratorWithGroupAdvantages", ["_compute_group_advantages"])
    obj.group = bind_methods(group_cls, SimpleNamespace(adv_method="grpo"), ["_compute_group_advantages"])
    token_cls = project_class("episode_generators/episode_generator_with_reward_function.py",
                               "EpisodeGeneratorWithRewardFunction", ["_tokenize_trajectory"])
    obj.tokenize = MethodType(token_cls._tokenize_trajectory, SimpleNamespace(tokenizer=obj.tokenizer))
    expander = project_class("inference_strategies/tree_inference/expansion.py", "EfficientIIDExpander", ["_compute_max_tokens"])
    obj.max_tokens = MethodType(expander._compute_max_tokens, SimpleNamespace(tokenizer=obj.tokenizer, model_context_size=200))
    # Load both real extraction classes together so super() retains its class cell.
    source = ROOT / "inference_strategies/tree_inference/answer_extraction.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name in ("IdentityAnswerExtractor", "IdentityWithSolutionPrefix")]
    for cls in classes:
        cls.decorator_list = []
    namespace = dict(AnswerExtractor=type("AnswerExtractor", (), {"__init__": lambda self, **kwargs: None}))
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    obj.extractor = namespace["IdentityWithSolutionPrefix"](**obj.protocol["answer_extractor"])
    return obj


def record(response="a\nb\n#### 1", row=7, eos=False):
    prompt = "Q\nSolution:"
    tokens = list(map(ord, response)) + ([0] if eos else [])
    rollout = dict(response_token_ids=tokens, response_text=response, reward=1, finish_reason="stop")
    return dict(row_index=row, source_index=row*10, selected_index=1, group_size=3,
                group_rewards=[0, 1, -2], gold_answer="1", question="Q", prompt_text=prompt,
                prompt_token_ids=list(map(ord, prompt)), selected_rollout=rollout)


class MethodTests(unittest.TestCase):
    def test_grpo_uses_project_population_std(self):
        methods = make_methods()
        result = methods.grpo_estimate([-2, 0, 1], 2, 4)
        self.assertAlmostEqual(result["group_std"], math.sqrt(14/9))
        self.assertAlmostEqual(result["token_advantages"][0], (4/3)/(math.sqrt(14/9)+1e-8))
        self.assertEqual(methods.grpo_estimate([1, 1], 0, 3)["token_advantages"], [0, 0, 0])

    def test_project_int5_and_zero_first_segment(self):
        methods = make_methods()
        r = record("abcdefghij")
        probs = [.99, .2, .99, .3, .4, .5, .6, .99, .7, .8]
        p = methods.prepare(r, [math.log(x) for x in probs])
        self.assertEqual(p["spo_int5"]["cutpoints"], [0, 7])
        self.assertEqual([q["query_text"] for q in p["spo_int5"]["requests"]],
                         [r["prompt_text"]+"a", r["prompt_text"]+"abcdefgh"])
        result = methods.estimate(p, "spo_int5", {0: dict(value=.25, std=0), 1: dict(value=.75, std=0)})
        self.assertEqual(result["token_advantages"], [0]+[.5]*7+[.25]*2)
        # High-probability interior tokens retain their advantage, as requested.
        self.assertEqual(result["token_advantages"][2], .5)

    def test_spo_no_cutpoints_no_mc_no_advantage(self):
        methods = make_methods()
        p = methods.prepare(record("abc"), [0.0]*3)
        self.assertEqual(p["spo_int5"]["requests"], [])
        self.assertEqual(methods.estimate(p, "spo_int5", {})["token_advantages"], [0, 0, 0])

    def test_spo_initial_baseline_negative_reward_and_default_mask(self):
        methods = make_methods()
        r = record("abcdefg")
        r["selected_rollout"]["reward"] = -2
        p = methods.prepare(r, [-2.0]*7)
        self.assertEqual(p["spo_int5"]["cutpoints"], [-1, 4])
        stats = {0: dict(value=.25, std=0), 1: dict(value=.75, std=0)}
        self.assertEqual(methods.estimate(p, "spo_int5", stats)["token_advantages"], [.5]*5+[-2.75]*2)
        traj = dict(p["trajectory"], cutpoints=[-1, 4], cutpoints_values=[.25, .75],
                    cutpoints_values_std=[0, 0], response_prob=[.99]*7)
        self.assertEqual(methods.spo._compute_token_advantages(traj)[1], [0]*7)
        self.assertEqual(methods.spo._compute_token_advantages(traj, apply_probability_mask=False)[1], [.5]*5+[-2.75]*2)

    def test_vine_character_boundary_is_not_snapped(self):
        class CrossingTokenizer(CharacterTokenizer):
            def __call__(self, text, **kwargs):
                pos = text.index("a\nb")
                ids = list(map(ord, text[:pos])) + [999] + list(map(ord, text[pos+3:]))
                offsets = [(i, i+1) for i in range(pos)] + [(pos, pos+3)] + [(i, i+1) for i in range(pos+3, len(text))]
                return dict(input_ids=ids, offset_mapping=offsets)
        methods = make_methods(CrossingTokenizer())
        r = record("a\nb\nc")
        r["selected_rollout"]["response_token_ids"] = [999, ord("\n"), ord("c")]
        p = methods.prepare(r, [-1.0]*3)
        self.assertEqual(p["vineppo"]["character_boundaries"], [0, 1, 3, 5])
        self.assertEqual(p["vineppo"]["token_step_indices"], [0, 2, 2])
        self.assertEqual([q["query_text"] for q in p["vineppo"]["requests"]],
                         [r["prompt_text"], r["prompt_text"]+"a", r["prompt_text"]+"a\nb"])
        result = methods.estimate(p, "vineppo", {0: dict(value=.1), 1: dict(value=.4), 2: dict(value=.6)})
        np.testing.assert_allclose(result["token_advantages"], [.3, .4, .4])
        self.assertEqual(len(result["segment_advantages"]), 3)  # Step with no token stays present.

    def test_vine_cap_and_project_tail_fill(self):
        methods = make_methods(cap=25)
        r = record("\n".join(["x"]*30))
        p = methods.prepare(r, [-1.0]*len(r["selected_rollout"]["response_token_ids"]))
        self.assertEqual(len(p["vineppo"]["steps"]), 30)
        self.assertEqual([q["value_index"] for q in p["vineppo"]["requests"]], list(range(26)))
        result = methods.estimate(p, "vineppo", {i: dict(value=.25) for i in range(26)})
        self.assertEqual(result["sampled_values"][26:], [None]*5)
        self.assertEqual(result["filled_values"][26:], [1, 1, 1, 1, 0])
        self.assertEqual(result["segment_advantages"][25:], [.75, 0, 0, 0, 0])

    def test_eos_mapping_and_noncanonical_ids_fail(self):
        methods = make_methods()
        r = record("abc", eos=True)
        p = methods.prepare(r, [-1.0]*3)
        self.assertEqual(p["reference_token_indices"], [0, 1, 2])
        self.assertEqual(p["excluded_reference_token_indices"], [3])
        validate_partitions(r, json.loads(json.dumps(p)), methods)
        r["selected_rollout"]["response_token_ids"][0] = 999
        with self.assertRaisesRegex(ValueError, "retokenization"):
            methods.base_trajectory(r)

    def test_project_horizon_and_dedup_preserve_destinations(self):
        methods = make_methods()
        r = record("abc")
        p = methods.prepare(r, [0.0]*3)
        # Preserve value indices even if two original character states have identical text.
        p["vineppo"]["requests"].append(dict(value_index=1, query_text=r["prompt_text"]))
        args = SimpleNamespace(rounds=1, mc_samples=9)
        meta = dict(seed=42, temperature=.6, top_p=.9, max_new_tokens=100, stop_text="STOP")
        jobs = list(build_jobs([r], [p], SimpleNamespace, meta, args, methods))
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["targets"], [(0, 0), (0, 1)])
        self.assertEqual(jobs[0]["max_tokens"], 100)
        self.assertEqual(jobs[0]["prompt_token_ids"][0], methods.tokenizer.bos_token_id)
        self.assertEqual(methods.max_tokens("x"*150, 100), 50)

    def test_scoring_uses_engine_text_and_project_extractor(self):
        methods = make_methods()
        r = record()
        completions = [SimpleNamespace(text="#### 1", finish_reason="stop"),
                       SimpleNamespace(text="#### 1", finish_reason="length"),
                       SimpleNamespace(text="#### 1\n#### 2", finish_reason="stop")]
        stat, children = methods.score_completions(r, "spo_int5", r["prompt_text"]+"prefix", completions)
        self.assertEqual(stat["reward_samples"], [1, 0, -2])
        self.assertEqual(children[0]["answer"], "prefix#### 1")
        self.assertAlmostEqual(stat["value"], -1/3)
        self.assertAlmostEqual(stat["std"], math.sqrt(14/9))

    def test_parallel_matches_serial_multiple_methods_rounds_rows(self):
        methods = make_methods()
        records = [record("abc", row=7), record("a\nb", row=3)]
        parts = [methods.prepare(r, [-1.0]*3) for r in records]
        meta = dict(seed=42, temperature=.6, top_p=.9, max_new_tokens=20, stop_text="STOP")
        outputs, audits = [], []
        for backend in ("serial", "engine"):
            args = SimpleNamespace(rounds=2, mc_samples=4, backend=backend, max_pending_requests=3)
            llm, audit = FakeLLM(), io.StringIO()
            llm.llm_engine.model_config = SimpleNamespace(max_model_len=201)
            result = list(compute_records(records, parts, llm, SimpleNamespace, methods, meta, args, audit))
            outputs.append(sorted(result, key=lambda r: r["row_index"]))
            audits.append(sorted([json.loads(s) for s in audit.getvalue().split("\n") if s],
                                 key=lambda r: (r["method"], r["round_index"], r["seed"])))
            self.assertFalse(llm.llm_engine.pending)
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(audits[0], audits[1])
        for r in outputs[0]:
            self.assertEqual(r["rounds"][0]["grpo"]["group_rewards"], [0, 1, -2])
            for rnd in r["rounds"]:
                for method in ("grpo", "vineppo", "spo_int5"):
                    self.assertEqual(len(rnd[method]["token_advantages"]), r["T"])
        self.assertEqual(sum(r["generated_token_count"] for r in outputs[0]),
                         sum(len(c["token_ids"]) for a in audits[0] for c in a["completions"]))

    def test_seeds_do_not_overlap_references(self):
        seeds = []
        for row in (0, 7):
            for rnd in (0, 1):
                for value_index in (0, 25):
                    seeds += [reference_seed(42, g, rnd, row, value_index) for g in (1, 2)]
                    seeds += [estimator_seed(42, m, rnd, row, value_index) for m in ("vineppo", "spo_int5", "grpo")]
        self.assertEqual(len(seeds), len(set(seeds)))


if __name__ == "__main__":
    unittest.main()
