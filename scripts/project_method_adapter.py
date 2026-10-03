"""Adapters to actual project estimators; the requested SPO mask is disabled."""
import asyncio
import copy
import hashlib
import json
import math
from pathlib import Path
from types import MethodType, SimpleNamespace

SCHEMA = "project_methods_v2"
ROOT = Path(__file__).resolve().parents[1]


def estimator_seed(base, method, round_index, row_index, value_index):
    from diagnostic_reference_utils import reference_seed
    coordinate = reference_seed(base, 1, round_index, row_index, value_index) - 2**48
    return {"vineppo": 3, "spo_int5": 4, "grpo": 5}[method] * 2**48 + coordinate


def load_project_protocol(seed=42):
    import _jsonnet
    def read(name):
        path = ROOT / "configs" / name
        expression = '''local c = import %s; {
          cap: c.episode_generator.max_step_for_value_estimation,
          delimiter: c.episode_generator.reasoning_step_delimiter,
          sampling: c.episode_generator.value_estimation_inference_strategy.node_expander.program_kwargs,
          context: c.episode_generator.value_estimation_inference_strategy.node_expander.model_context_size,
          samples: c.episode_generator.value_estimation_inference_strategy.samples,
          extractor: c.episode_generator.value_estimation_inference_strategy.answer_extractor,
          task: c.episode_generator.task,
        }''' % json.dumps(path.as_posix())
        return json.loads(_jsonnet.evaluate_snippet("method_protocol", expression,
                          ext_vars={"APP_SEED": str(seed), "APP_OPENAI_VLLM_API_BASE": "none"}))
    vine = read("polIter_rho1bSft2_vineppo_GSM8K.jsonnet")
    spo = read("polIter_rho1bSft2_spo_chain_GSM8K.jsonnet")
    for key in ("sampling", "context", "samples", "extractor", "task", "delimiter"):
        if vine[key] != spo[key]:
            raise ValueError(f"Project configs disagree on {key}")
    interval = json.loads(_jsonnet.evaluate_file(str(ROOT / "configs/episode_generators/interval5.jsonnet")))
    return dict(schema=SCHEMA, vine_max_steps=vine["cap"], delimiter=vine["delimiter"],
                sampling=vine["sampling"], model_context_size=vine["context"], mc_samples=vine["samples"],
                answer_extractor=vine["extractor"], task=vine["task"],
                spo_interval=interval["episode_generator"]["cutpoint_interval"], probability_mask=False,
                advantage_stage="episode_generator_before_trainer_whitening")


def source_fingerprint():
    paths = sorted((ROOT / "configs").rglob("*.jsonnet"))
    paths += [ROOT / ("src/treetune/" + p) for p in (
        "tasks/gsm8k.py", "episode_generators/episode_generator_with_reward_function.py",
        "episode_generators/vineppo_episode_generator.py",
        "episode_generators/math_episode_generator_with_mc_advantages.py",
        "episode_generators/math_episode_generator_with_group_advantages.py",
        "episode_generators/math_episode_generator.py", "trainers/ppo_trainer.py",
        "inference_strategies/tree_inference/expansion.py",
        "inference_strategies/tree_inference/answer_extraction.py")]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_sampling_metadata(metadata, protocol):
    for name, config_name in (("temperature", "temperature"), ("top_p", "top_p"),
                              ("max_new_tokens", "max_tokens")):
        if metadata[name] != protocol["sampling"][config_name]:
            raise ValueError(f"Trajectory {name} differs from project config")
    # Jsonnet already expands \n escapes. The remaining quotes belong to the
    # guidance expression, whose string may contain literal control characters.
    # Keep those characters exactly; JSON's default strict=True rejects them.
    if metadata["stop_text"] != json.loads(protocol["sampling"]["stop"], strict=False):
        raise ValueError("Trajectory stop text differs from project config")


def bind_methods(cls, context, names):
    for name in names:
        setattr(context, name, MethodType(getattr(cls, name), context))
    return context


class ProjectMethods:
    def __init__(self, tokenizer, protocol):
        from diagnostic_reference_utils import project_reward_function
        reward = project_reward_function()
        from treetune.episode_generators.vineppo_episode_generator import VinePPOEpisodeGenerator
        from treetune.episode_generators.math_episode_generator_with_mc_advantages import MathEpisodeGeneratorWithMCAdvantages
        from treetune.episode_generators.math_episode_generator_with_group_advantages import MathEpisodeGeneratorWithGroupAdvantages
        from treetune.episode_generators.episode_generator_with_reward_function import EpisodeGeneratorWithRewardFunction
        from treetune.inference_strategies.tree_inference.expansion import EfficientIIDExpander
        from treetune.inference_strategies.tree_inference.answer_extraction import IdentityWithSolutionPrefix
        self.tokenizer, self.protocol, self.task = tokenizer, protocol, reward.math_task
        self.vine_cls = VinePPOEpisodeGenerator
        self.vine = bind_methods(VinePPOEpisodeGenerator, SimpleNamespace(
            tokenizer=tokenizer, reward_function=reward, reasoning_step_delimiter=protocol["delimiter"],
            max_step_for_value_estimation=protocol["vine_max_steps"]), (
                "_create_step_query", "_get_value_estimation_queries", "_compute_step_advantages",
                "_compute_token_advantages", "_compute_mc_value"))
        self.spo = bind_methods(MathEpisodeGeneratorWithMCAdvantages, SimpleNamespace(
            tokenizer=tokenizer, reward_function=reward, cutpoint_interval=protocol["spo_interval"]), (
                "_get_response_probs_and_cutpoints", "_create_cutpoint_query",
                "_compute_token_advantages", "_compute_mc_value"))
        self.group = bind_methods(MathEpisodeGeneratorWithGroupAdvantages,
                                  SimpleNamespace(adv_method="grpo"), ("_compute_group_advantages",))
        self.tokenize = MethodType(EpisodeGeneratorWithRewardFunction._tokenize_trajectory,
                                  SimpleNamespace(tokenizer=tokenizer))
        self.max_tokens = MethodType(EfficientIIDExpander._compute_max_tokens, SimpleNamespace(
            tokenizer=tokenizer, model_context_size=protocol["model_context_size"]))
        self.extractor = IdentityWithSolutionPrefix(**{k: v for k, v in protocol["answer_extractor"].items()
                                                       if k != "type"})

    def base_trajectory(self, record):
        rollout = record["selected_rollout"]
        traj = dict(query_text=record["prompt_text"], response_text=rollout["response_text"],
                    full_text=record["prompt_text"] + rollout["response_text"], score=rollout["reward"])
        query, response, offsets = self.tokenize(traj, is_unfinished_response=rollout["finish_reason"] == "length",
                                                 return_offsets=True)
        original = rollout["response_token_ids"]
        if query != record["prompt_token_ids"] or not response:
            raise ValueError("Project tokenization changed the prompt or produced an empty response")
        if response != original and not (original[-1] == self.tokenizer.eos_token_id and response == original[:-1]):
            raise ValueError("Project retokenization differs from saved IDs; cannot align existing references")
        traj.update(query_token_ids=query, response_token_ids=response,
                    offsets=[list(pair) for pair in offsets])
        return traj

    def vine_partition(self, traj):
        traj = copy.deepcopy(traj)
        indices = self.task.split_solution_into_intermediate_steps(traj["response_text"])
        traj["step_indices"] = indices
        traj["steps"] = [traj["response_text"][a:b] for a, b in zip(indices, indices[1:])]
        queries = self.vine._get_value_estimation_queries(traj)
        context = SimpleNamespace(tokenizer=self.tokenizer, reasoning_step_delimiter=self.protocol["delimiter"],
                                  _compute_step_advantages=lambda _: list(range(len(traj["steps"]))))
        token_steps = self.vine_cls._compute_token_advantages(context, traj)
        return dict(character_boundaries=indices, steps=traj["steps"], token_step_indices=token_steps,
                    requests=[dict(value_index=i, query_text=q) for i, q in queries])

    def spo_partition(self, traj, logprobs):
        if len(logprobs) != len(traj["response_token_ids"]) or any(
                not math.isfinite(p) or p > 1e-6 for p in logprobs):
            raise ValueError("Invalid actor logprobs")
        probs, cuts = self.spo._get_response_probs_and_cutpoints(logprobs)
        return dict(cutpoints=cuts.tolist(), response_prob=probs.tolist(),
                    requests=[dict(value_index=i, query_text=self.spo._create_cutpoint_query(traj, int(c)))
                              for i, c in enumerate(cuts)])

    def prepare(self, record, logprobs):
        traj = self.base_trajectory(record)
        original = record["selected_rollout"]["response_token_ids"]
        return dict(schema=SCHEMA, row_index=record["row_index"], source_index=record["source_index"],
                    selected_index=record["selected_index"], trajectory=traj,
                    original_response_token_ids=original,
                    reference_token_indices=list(range(len(traj["response_token_ids"]))),
                    excluded_reference_token_indices=list(range(len(traj["response_token_ids"]), len(original))),
                    actor_token_logprobs=logprobs, vineppo=self.vine_partition(traj),
                    spo_int5=self.spo_partition(traj, logprobs))

    def grpo_estimate(self, rewards, selected_index, length):
        import numpy as np
        if len(rewards) < 2 or not all(math.isfinite(r) for r in rewards):
            raise ValueError("Invalid GRPO rewards")
        trajectories = [dict(instance_idx=0, score=r) for r in rewards]
        self.group._compute_group_advantages(trajectories)
        mean, std = float(np.mean(rewards)), float(np.std(rewards))
        return dict(group_rewards=rewards, group_mean=mean, group_std=std, epsilon=1e-8, std_ddof=0,
                    token_advantages=[float(trajectories[selected_index]["advantage"])] * length,
                    token_advantages_unstandardized=[rewards[selected_index]-mean] * length)

    def estimate(self, prepared, method, statistics):
        traj = copy.deepcopy(prepared["trajectory"])
        partition = prepared[method]
        if set(statistics) != {r["value_index"] for r in partition["requests"]}:
            raise ValueError("Missing or unexpected project value requests")
        if method == "vineppo":
            count = len(partition["steps"])
            traj.update(step_indices=partition["character_boundaries"], steps=partition["steps"],
                        step_rewards=[0.0]*(count-1)+[traj["score"]], values=[None]*(count+1))
            for index, stat in statistics.items():
                traj["values"][index] = stat["value"]
            raw_values = list(traj["values"])
            steps = self.vine._compute_step_advantages(copy.deepcopy(traj))
            advantages = self.vine._compute_token_advantages(traj)
            return dict(character_boundaries=traj["step_indices"], token_step_indices=partition["token_step_indices"],
                        sampled_values=raw_values, filled_values=traj["values"], segment_advantages=steps,
                        token_advantages=[float(a) for a in advantages])
        if method != "spo_int5":
            raise ValueError(method)
        traj.update(cutpoints=partition["cutpoints"], response_prob=partition["response_prob"],
                    cutpoints_values=[statistics[i]["value"] for i in range(len(partition["cutpoints"]))],
                    cutpoints_values_std=[statistics[i]["std"] for i in range(len(partition["cutpoints"]))])
        token_values, advantages = self.spo._compute_token_advantages(traj, apply_probability_mask=False)
        return dict(cutpoints=traj["cutpoints"], cutpoint_values=traj["cutpoints_values"],
                    token_values=token_values, token_advantages=[float(a) for a in advantages])

    def score_completions(self, record, method, query_text, completions):
        children = []
        for completion in completions:
            if completion.finish_reason not in ("stop", "length"):
                raise ValueError(f"Unexpected finish reason: {completion.finish_reason}")
            answer = completion.text if method == "grpo" else asyncio.run(self.extractor.extract_from_node(
                dict(full_text=query_text + completion.text, finish_reason=completion.finish_reason)))
            children.append(dict(answer=answer, finish_reason=completion.finish_reason))
        context = self.spo if method == "spo_int5" else self.vine
        result = context._compute_mc_value(query=query_text,
                    value_estimation_result={"_treetune__reasoning_tree": json.dumps({"children": children})},
                    data_instance={"problem": record["question"], "answer": record["gold_answer"]}, return_rewards=True)
        if method == "spo_int5":
            value, std, rewards = result
        else:
            value, rewards = result
            std = None
        return dict(value=float(value), std=None if std is None else float(std),
                    reward_samples=[float(r) for r in rewards]), children
