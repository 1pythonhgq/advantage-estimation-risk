"""Generate strict diagnostic trajectories with the project's vLLM engine.

This script deliberately does not import or call ``transformers.generate``.
It uses the vLLM version installed by this repository and submits tokenized
prompts through vLLM's ``prompt_token_ids`` interface.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_from_disk
from vllm import LLM, SamplingParams

from diagnostic_reference_utils import REWARD_PROTOCOL, project_reward_function, score_response


# Must match configs/prompt_library/MATH_step_by_step_sft.jsonnet
# (prompt_library.tree.question_template), which is inherited by GSM8K.
DEFAULT_PROMPT_TEMPLATE = "[MATH_TASK] Problem:\n{query}\n\nSolution:"
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_GROUP_SIZE = 8
DEFAULT_SEED = 42


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GSM8K trajectories with vLLM prompt-token input."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/gsm8k_validation_selection/pilot"),
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/gsm8k_validation_selection/trajectories_vllm.jsonl"),
    )
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS
    )
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--stop-text", default="\n\n\nProblem:")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--logprobs",
        type=int,
        default=1,
        help="Number of top logprobs requested from vLLM; the chosen token must be present.",
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument(
        "--disable-prefix-caching",
        action="store_true",
        help="Leave enabled by default, matching the GSM8K training configs.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--allow-text-token-mismatch",
        action="store_true",
        help="Debug-only escape hatch; strict experiments should leave this disabled.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_dataset(path: Path) -> Dataset:
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    dataset = load_from_disk(str(path))
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected Dataset at {path}, got {type(dataset).__name__}")
    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty: {path}")
    return dataset


def _get_question(row: Dict[str, Any]) -> str:
    for field in ("query", "problem", "question"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise KeyError("Dataset row must contain query, problem, or question")


def _get_gold_answer(row: Dict[str, Any]) -> str:
    value = row.get("_final_answer")
    if isinstance(value, str) and value.strip():
        return value.strip()
    value = row.get("answer")
    if not isinstance(value, str):
        raise KeyError("Dataset row must contain answer or _final_answer")
    return value.rsplit("####", 1)[-1].strip()


def _score_response(response_text: str, gold_answer: str, finish_reason: str) -> float:
    return score_response(response_text, gold_answer, finish_reason)


def _extract_logprobs(completion: Any, token_ids: List[int]) -> List[float]:
    """Extract chosen-token logprobs and fail loudly if one is missing."""
    if not token_ids:
        return []
    values = getattr(completion, "logprobs", None)
    if values is None:
        raise ValueError(
            "vLLM returned no logprobs. Check SamplingParams(logprobs=...)."
        )
    if len(values) != len(token_ids):
        raise ValueError(
            "vLLM logprob length mismatch: "
            f"{len(values)} entries for {len(token_ids)} response tokens."
        )

    output: List[float] = []
    for position, (token_id, candidates) in enumerate(zip(token_ids, values)):
        if candidates is None:
            raise ValueError(
                f"Missing vLLM logprobs at response position {position} "
                f"for token id {token_id}."
            )
        candidate = candidates.get(token_id)
        if candidate is None:
            # Some vLLM builds expose token ids as strings in this mapping.
            candidate = candidates.get(str(token_id))
        if candidate is None:
            returned_ids = list(candidates.keys())
            raise ValueError(
                "The sampled token is absent from vLLM logprobs at "
                f"response position {position}: sampled={token_id}, "
                f"returned={returned_ids[:20]}. Increase --logprobs or verify "
                "the installed vLLM behavior."
            )
        elif hasattr(candidate, "logprob"):
            value = float(candidate.logprob)
        elif isinstance(candidate, dict) and "logprob" in candidate:
            value = float(candidate["logprob"])
        else:
            value = float(candidate)
        if not math.isfinite(value):
            raise ValueError(
                "vLLM returned a non-finite sampled-token logprob at "
                f"response position {position}: {value!r}."
            )
        output.append(value)
    return output


def generate_trajectories(
    dataset_path: Path,
    model_name_or_path: str,
    output_path: Path,
    group_size: int = DEFAULT_GROUP_SIZE,
    limit: Optional[int] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    stop_text: str = "\n\n\nProblem:",
    seed: int = DEFAULT_SEED,
    logprobs: int = 1,
    dtype: str = "auto",
    gpu_memory_utilization: float = 0.5,
    max_model_len: Optional[int] = None,
    enable_prefix_caching: bool = True,
    trust_remote_code: bool = False,
    allow_text_token_mismatch: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if logprobs <= 0:
        raise ValueError("logprobs must be positive")
    if "{query}" not in prompt_template:
        raise ValueError("prompt_template must contain {query}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {output_path}; pass --overwrite to replace it."
        )

    project_reward_function()
    dataset = _load_dataset(dataset_path)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    llm_kwargs: Dict[str, Any] = {
        "model": model_name_or_path,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enable_prefix_caching": enable_prefix_caching,
        "trust_remote_code": trust_remote_code,
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.with_suffix(".meta.json")
    records: List[Dict[str, Any]] = []
    for row_index, row in enumerate(dataset):
        question = _get_question(row)
        gold_answer = _get_gold_answer(row)
        prompt_text = prompt_template.format(query=question)
        prompt_token_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        question_seed = seed + row_index
        sampling_kwargs: Dict[str, Any] = {
            "n": group_size,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
            "logprobs": logprobs,
            "seed": question_seed,
        }
        if stop_text:
            sampling_kwargs["stop"] = [stop_text]
            sampling_kwargs["include_stop_str_in_output"] = False
        sampling_params = SamplingParams(**sampling_kwargs)
        request_outputs = llm.generate(
            prompt_token_ids=[prompt_token_ids],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        if len(request_outputs) != 1:
            raise RuntimeError(f"Expected one vLLM request output, got {len(request_outputs)}")

        completions = request_outputs[0].outputs
        if len(completions) != group_size:
            raise RuntimeError(
                f"Expected {group_size} completions, got {len(completions)}"
            )

        rollouts: List[Dict[str, Any]] = []
        for completion in completions:
            response_token_ids = list(completion.token_ids)
            response_text = completion.text
            finish_reason = completion.finish_reason or "length"
            decoded_response = tokenizer.decode(
                response_token_ids,
                # Keep special tokens visible so an unexpected EOS or other
                # token cannot be hidden by the validation itself.
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            decoded_response_without_special = tokenizer.decode(
                response_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            # vLLM may omit a terminal special token from ``completion.text``
            # while retaining it in ``token_ids``.  Accept that one documented
            # representation difference, but reject any ordinary text/token
            # mismatch (including a stop-string boundary mismatch).
            text_token_match = (
                decoded_response == response_text
                or decoded_response_without_special == response_text
            )
            if not text_token_match:
                message = (
                    "vLLM text/token mismatch at row "
                    f"{row_index}: decoded={decoded_response!r}, "
                    f"decoded_without_special={decoded_response_without_special!r}, "
                    f"returned_text={response_text!r}"
                )
                if not allow_text_token_mismatch:
                    raise ValueError(message)
                print(f"WARNING: {message}")
            token_logprobs = _extract_logprobs(completion, response_token_ids)
            rollouts.append(
                {
                    "response_token_ids": response_token_ids,
                    "token_logprobs": token_logprobs,
                    "response_text": response_text,
                    "finish_reason": finish_reason,
                    "reward": _score_response(
                        response_text, gold_answer, finish_reason
                    ),
                }
            )

        selected_index = random.Random(seed + 1_000_000 + row_index).randrange(
            group_size
        )
        unique_response_count = len({rollout["response_text"] for rollout in rollouts})
        if unique_response_count == 1 and group_size > 1:
            print(
                f"WARNING: all {group_size} samples are identical at row "
                f"{row_index}; verify vLLM seed/n sampling behavior."
            )
        source_index = row.get("source_index", row.get("_treetune__idx", row_index))
        records.append(
            {
                "row_index": row_index,
                "source_index": source_index,
                "question": question,
                "gold_answer": gold_answer,
                "prompt_text": prompt_text,
                "prompt_token_ids": prompt_token_ids,
                "group_size": group_size,
                "selected_index": selected_index,
                "unique_response_count": unique_response_count,
                "group_rewards": [rollout["reward"] for rollout in rollouts],
                "selected_rollout": rollouts[selected_index],
                "rollouts": rollouts,
            }
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    try:
        import vllm

        vllm_version = getattr(vllm, "__version__", "unknown")
    except Exception:
        vllm_version = "unknown"
    metadata = {
        "dataset_path": str(dataset_path.resolve()),
        "model_name_or_path": model_name_or_path,
        "inference_engine": "vllm.LLM",
        "reward_protocol": REWARD_PROTOCOL,
        "vllm_version": vllm_version,
        "num_questions": len(records),
        "limit": limit,
        "group_size": group_size,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "prompt_template": prompt_template,
        "stop_text": stop_text,
        "seed": seed,
        "logprobs": logprobs,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "trust_remote_code": trust_remote_code,
        "engine_seed": llm.llm_engine.model_config.seed,
        "enable_prefix_caching": enable_prefix_caching,
        "prompt_mode": "prompt_token_ids",
        "random_streams": {
            "trajectory_generation": "seed + row_index",
            "trajectory_selection": "seed + 1000000 + row_index",
        },
        "selection_random_stream_offset": 1_000_000,
        "output_path": str(output_path.resolve()),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return metadata


def main() -> None:
    args = _parse_args()
    metadata = generate_trajectories(
        dataset_path=args.dataset_path,
        model_name_or_path=args.model_name_or_path,
        output_path=args.output_path,
        group_size=args.group_size,
        limit=args.limit,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        prompt_template=args.prompt_template,
        stop_text=args.stop_text,
        seed=args.seed,
        logprobs=args.logprobs,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_prefix_caching=not args.disable_prefix_caching,
        trust_remote_code=args.trust_remote_code,
        allow_text_token_mismatch=args.allow_text_token_mismatch,
        overwrite=args.overwrite,
    )
    print(
        f"Generated {metadata['num_questions']} groups with vLLM "
        f"(G={metadata['group_size']}) -> {metadata['output_path']}"
    )


if __name__ == "__main__":
    main()
