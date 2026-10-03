"""Project reward adapter and helpers for two-group MC reference advantages."""

import math
import statistics
import sys
from functools import lru_cache
from pathlib import Path


REWARD_PROTOCOL = {
    "name": "gsm8k_project_terminal_v1",
    "implementation": "treetune.episode_generators.math_episode_generator.MATHRewardFunction",
    "task": "GSM8K(use_original_format=True)",
    "penalize_unfinished_response": True,
    "unfinished_response_penalty": 0.0,
    "length_finish_reward": 0.0,
    "multiple_answer_penalty": -2.0,
    "answer_normalization": "prediction: strip/remove commas/lower; gold: strip/lower",
}


@lru_cache(maxsize=1)
def project_reward_function():
    """Import the repository implementation, rather than duplicate its grader.

    Constructing the task does not load its dataset. tokenizer is unused by
    the GSM8K reward methods; no model or GPU is needed for scoring.
    """
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from treetune.tasks.gsm8k import GSM8K
    from treetune.episode_generators.math_episode_generator import MATHRewardFunction

    task = GSM8K(use_original_format=True, remove_calculator_expressions=True,
                 load_dataset_dict=True, dataset_dict_path="data/gsm8k")
    return MATHRewardFunction(tokenizer=None, math_task=task,
                              penalize_unfinished_response=True,
                              unfinished_response_penalty=0.0)


def score_response(text, gold, finish_reason, reward_mode="project"):
    """Use the GSM8K SPO-chain/VinePPO terminal and MC reward rule.

    ``trajectory`` remains an alias for older commands, never a second rule.
    No KL shaping or advantage normalization is included in this task reward.
    """
    if reward_mode not in ("project", "trajectory"):
        raise ValueError("Use reward_mode='project'; binary rewards differ from training")
    if finish_reason not in ("stop", "length"):
        raise ValueError(f"Unexpected finish reason: {finish_reason!r}")
    reward_function = project_reward_function()
    if finish_reason == "length":
        return reward_function.get_unfinished_response_penalty()
    # In original-format GSM8K the problem/query arguments are not used by
    # extraction or grading. gold is the final answer, not the full solution.
    reward, _ = reward_function("", text, {"problem": "", "answer": gold})
    return float(reward)


def reference_seed(base, group, round_index, row_index, prefix_length):
    """Injective request coordinates in reserved 64-bit reference namespaces.

    Group 1/2 occupy disjoint ranges far above the existing generation and
    selection seeds. Reserve these ranges for reference sampling only.
    """
    for name, value, bound in (
        ("base", base, 2**32), ("round", round_index, 2**10),
        ("row", row_index, 2**20), ("prefix", prefix_length, 2**16),
    ):
        if not isinstance(value, int) or not 0 <= value < bound:
            raise ValueError(f"{name} must be an integer in [0, {bound})")
    if group not in (1, 2):
        raise ValueError("Reference group must be 1 or 2")
    return (group * 2**48 + base + round_index * 2**36
            + row_index * 2**16 + prefix_length)


def summarize_reference(reward_samples, terminal_reward):
    """Prefix i contains i response tokens; advantage i belongs to token i."""
    if not reward_samples or any(len(samples) < 2 for samples in reward_samples):
        raise ValueError("Need all nonterminal prefixes and K >= 2 samples each")
    if len({len(samples) for samples in reward_samples}) != 1:
        raise ValueError("Every prefix must have the same MC budget")
    if not all(math.isfinite(x) for s in reward_samples for x in s):
        raise ValueError("Non-finite rollout reward")
    values = [statistics.mean(s) for s in reward_samples] + [0.0]
    variances = [statistics.variance(s) / len(s) for s in reward_samples] + [0.0]
    advantages = [values[i + 1] - values[i] for i in range(len(reward_samples))]
    advantages[-1] += terminal_reward
    residual = math.fsum(advantages) - (terminal_reward - values[0])
    if not math.isfinite(residual) or abs(residual) > 1e-10:
        raise ValueError(f"Telescoping identity failed: {residual}")
    return {
        "values": values,
        "value_standard_errors": [math.sqrt(v) for v in variances],
        "token_advantages": advantages,
        # Adjacent advantages share a value estimate and are correlated.
        "token_standard_errors": [math.sqrt(variances[i] + variances[i + 1])
                                  for i in range(len(advantages))],
        "telescoping_residual": residual,
    }


def completed_response(tokenizer, prompt_ids, response_ids, stop_text, finish_reason):
    """Decode in context for scoring only; never re-encode a sampled prefix.

    Searching the complete response also detects a stop string straddling
    the fixed-prefix / newly-generated-token boundary (vLLM only checks its
    generated text). Raw token ids, including stop tokens, remain auditable.
    """
    if finish_reason not in ("stop", "length"):
        raise ValueError(f"Unexpected finish reason: {finish_reason!r}")
    kwargs = dict(skip_special_tokens=True, clean_up_tokenization_spaces=False)
    prompt_text = tokenizer.decode(prompt_ids, **kwargs)
    full_text = tokenizer.decode(prompt_ids + response_ids, **kwargs)
    if not full_text.startswith(prompt_text):
        raise ValueError("Prompt/response decoding boundary changed")
    text = full_text[len(prompt_text):]
    stop_position = text.find(stop_text) if stop_text else -1
    if stop_position >= 0:
        return text[:stop_position], "stop"
    return text, finish_reason
