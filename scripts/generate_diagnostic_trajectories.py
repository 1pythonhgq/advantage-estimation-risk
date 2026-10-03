"""Generate non-strict Transformers diagnostic trajectories for GSM8K.

For the final experiment, use ``generate_diagnostic_trajectories_vllm.py``.
This file is retained only for local smoke tests and does not use the same
inference engine as the training code.

The input dataset is produced by ``select_gsm8k_validation.py``.  For each
question, this script samples a group of answers with the same policy and
keeps the complete token ids of every answer.  One answer is selected with a
separate deterministic random stream as the diagnostic trajectory.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPT_TEMPLATE = "[MATH_TASK] Problem:\n{query}\n\nSolution:"
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_GROUP_SIZE = 8
DEFAULT_SEED = 42


@dataclass
class Rollout:
    response_token_ids: List[int]
    token_logprobs: List[float]
    response_text: str
    finish_reason: str
    reward: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GSM8K diagnostic trajectories from a causal LM."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/gsm8k_validation_selection/pilot"),
        help="A Dataset saved by select_gsm8k_validation.py.",
    )
    parser.add_argument(
        "--model-name-or-path",
        required=True,
        help="Hugging Face model name or local checkpoint path.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/gsm8k_validation_selection/trajectories.jsonl"),
        help="JSONL path for generated trajectories.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Optional metadata JSON path; defaults to output path with .meta.json.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=DEFAULT_GROUP_SIZE,
        help=f"Number of answers sampled per question (default: {DEFAULT_GROUP_SIZE}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N rows, useful for the two-question smoke test.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE}).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help=f"Nucleus sampling probability (default: {DEFAULT_TOP_P}).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Response token limit (default: {DEFAULT_MAX_NEW_TOKENS}).",
    )
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Prompt template containing {query}.",
    )
    parser.add_argument(
        "--stop-text",
        default="\n\n\nProblem:",
        help="Optional stop text to remove from the response; use an empty string to disable.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Base seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, for example auto, cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Model dtype (default: auto).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing output files.",
    )
    return parser.parse_args()


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _resolve_dtype(dtype_name: str, device: torch.device) -> Optional[torch.dtype]:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def _load_dataset(path: Path) -> Dataset:
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    dataset = load_from_disk(str(path))
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected a Dataset at {path}, got {type(dataset).__name__}")
    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty: {path}")
    return dataset


def _get_question(row: Dict[str, Any]) -> str:
    for field in ("query", "problem", "question"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise KeyError("Dataset row must contain one of: query, problem, question")


def _get_gold_answer(row: Dict[str, Any]) -> str:
    value = row.get("_final_answer")
    if isinstance(value, str) and value.strip():
        return value.strip()

    value = row.get("answer")
    if not isinstance(value, str):
        raise KeyError("Dataset row must contain a string answer or _final_answer")
    if "####" in value:
        value = value.rsplit("####", 1)[1]
    return value.strip()


def _score_response(response_text: str, gold_answer: str, finish_reason: str) -> float:
    # This matches the GSM8K reward convention used by math_reward_function:
    # correct answer = 1, all other or unfinished answers = 0, and multiple
    # conclusions receive -2.
    if finish_reason == "length":
        return 0.0
    parts = response_text.split("####")
    if len(parts) > 2:
        return -2.0
    if len(parts) < 2:
        return 0.0
    predicted = parts[-1].strip().replace(",", "").lower()
    expected = gold_answer.strip().replace(",", "").lower()
    return 1.0 if predicted == expected else 0.0


def _find_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> int:
    if not subsequence or len(subsequence) > len(sequence):
        return -1
    end = len(sequence) - len(subsequence) + 1
    for start in range(end):
        if list(sequence[start : start + len(subsequence)]) == list(subsequence):
            return start
    return -1


def _truncate_at_stop(
    response_token_ids: List[int], tokenizer, stop_text: str
) -> List[int]:
    if not stop_text:
        return response_token_ids
    stop_ids = tokenizer(
        stop_text, add_special_tokens=False, return_attention_mask=False
    )["input_ids"]
    stop_start = _find_subsequence(response_token_ids, stop_ids)
    if stop_start < 0:
        return response_token_ids
    return response_token_ids[:stop_start]


def _compute_token_logprobs(
    model,
    full_sequences: torch.Tensor,
    prompt_length: int,
    response_lengths: List[int],
    device: torch.device,
) -> List[List[float]]:
    """Compute raw policy log-probabilities for generated response tokens."""
    attention_mask = torch.ones_like(full_sequences, device=device)
    with torch.inference_mode():
        logits = model(
            input_ids=full_sequences,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits

    # Logits at position i predict token i + 1. The first response token is
    # therefore predicted by the logit at prompt_length - 1.
    response_start = prompt_length - 1
    response_end = response_start + max(response_lengths, default=0)
    response_logits = logits[:, response_start:response_end, :].float()
    response_logprobs = torch.log_softmax(response_logits, dim=-1)
    target_tokens = full_sequences[:, prompt_length:response_end + 1]
    gathered = response_logprobs.gather(
        dim=-1, index=target_tokens.unsqueeze(-1)
    ).squeeze(-1)
    return [
        gathered[row_idx, :response_length].detach().cpu().tolist()
        for row_idx, response_length in enumerate(response_lengths)
    ]


def _decode_response(tokenizer, response_token_ids: Iterable[int]) -> str:
    return tokenizer.decode(
        list(response_token_ids),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _generate_group(
    model,
    tokenizer,
    prompt_token_ids: List[int],
    gold_answer: str,
    group_size: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    stop_text: str,
    seed: int,
    device: torch.device,
) -> List[Rollout]:
    _set_seed(seed)
    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
    input_ids = input_ids.repeat(group_size, 1)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=False,
        )

    prompt_length = len(prompt_token_ids)
    raw_response_lengths: List[int] = []
    for sequence in generated.detach().cpu().tolist():
        raw_response = sequence[prompt_length:]
        if tokenizer.eos_token_id in raw_response:
            raw_response_lengths.append(
                raw_response.index(tokenizer.eos_token_id) + 1
            )
        else:
            raw_response_lengths.append(len(raw_response))
    raw_logprobs = _compute_token_logprobs(
        model=model,
        full_sequences=generated,
        prompt_length=prompt_length,
        response_lengths=raw_response_lengths,
        device=device,
    )

    rollouts: List[Rollout] = []
    for row_idx, sequence in enumerate(generated.detach().cpu().tolist()):
        raw_response_ids = sequence[prompt_length : prompt_length + raw_response_lengths[row_idx]]
        raw_response_logprobs = raw_logprobs[row_idx]
        response_ids = list(raw_response_ids)
        finish_reason = "eos" if tokenizer.eos_token_id in response_ids else "length"
        if tokenizer.eos_token_id in response_ids:
            eos_position = response_ids.index(tokenizer.eos_token_id)
            response_ids = response_ids[: eos_position + 1]
        stop_trimmed_ids = _truncate_at_stop(response_ids, tokenizer, stop_text)
        if len(stop_trimmed_ids) < len(response_ids):
            finish_reason = "stop"
        response_ids = stop_trimmed_ids
        response_logprobs = raw_response_logprobs[: len(response_ids)]
        response_text = _decode_response(tokenizer, response_ids)
        reward = _score_response(response_text, gold_answer, finish_reason)
        rollouts.append(
            Rollout(
                response_token_ids=response_ids,
                token_logprobs=response_logprobs,
                response_text=response_text,
                finish_reason=finish_reason,
                reward=reward,
            )
        )
    return rollouts


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
    device_name: str = "auto",
    dtype_name: str = "auto",
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Generate and save one grouped rollout record per input question."""
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero when sampling")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if "{query}" not in prompt_template:
        raise ValueError("prompt_template must contain {query}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output_path}. "
            "Pass --overwrite to replace it."
        )

    dataset = _load_dataset(dataset_path)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    device = _resolve_device(device_name)
    dtype = _resolve_dtype(dtype_name, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs either pad_token_id or eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

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
        rollouts = _generate_group(
            model=model,
            tokenizer=tokenizer,
            prompt_token_ids=prompt_token_ids,
            gold_answer=gold_answer,
            group_size=group_size,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            stop_text=stop_text,
            seed=question_seed,
            device=device,
        )
        selected_index = random.Random(seed + 1_000_000 + row_index).randrange(
            group_size
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
                "group_rewards": [rollout.reward for rollout in rollouts],
                "selected_rollout": asdict(rollouts[selected_index]),
                "rollouts": [asdict(rollout) for rollout in rollouts],
            }
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata = {
        "dataset_path": str(dataset_path.resolve()),
        "model_name_or_path": model_name_or_path,
        "num_questions": len(records),
        "limit": limit,
        "group_size": group_size,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "prompt_template": prompt_template,
        "stop_text": stop_text,
        "seed": seed,
        "device": str(device),
        "dtype": str(dtype),
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
        device_name=args.device,
        dtype_name=args.dtype,
        overwrite=args.overwrite,
    )
    print(
        f"Generated {metadata['num_questions']} question groups with "
        f"G={metadata['group_size']} and saved {metadata['output_path']}"
    )


if __name__ == "__main__":
    main()
