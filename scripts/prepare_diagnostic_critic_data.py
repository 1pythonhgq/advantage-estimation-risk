"""Prepare question-disjoint Monte Carlo return regression data for the critic."""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

from diagnostic_critic_utils import (gold_answer, question_key, question_split, question_text,
                                     read_jsonl, sha256, write_json)
from diagnostic_reference_utils import REWARD_PROTOCOL, project_reward_function, score_response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts-path", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-metadata-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Explicitly allow a subset of the critic pool (>=2 questions)")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh output directory")
    meta_path = args.rollouts_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    diagnostic = json.loads(args.diagnostic_metadata_path.read_text(encoding="utf-8"))
    if meta["inference_engine"] != "vllm.LLM" or meta["reward_protocol"] != REWARD_PROTOCOL:
        raise ValueError("Critic data must use the existing vLLM/project reward pipeline")
    for key in ("model_name_or_path", "inference_engine", "vllm_version", "temperature", "top_p",
                "max_new_tokens", "prompt_template", "stop_text", "reward_protocol"):
        if key not in diagnostic or meta[key] != diagnostic[key]:
            raise ValueError(f"Critic and diagnostic sampling protocols differ: {key}")
    manifest_path = args.selection_dir / "manifest.json"
    split_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pool_path, diagnostic_path = args.selection_dir / "critic.jsonl", args.selection_dir / "diagnostic.jsonl"
    pool, diagnostic_pool = read_jsonl(pool_path), read_jsonl(diagnostic_path)
    allowed = {r["source_index"]: r for r in pool}
    forbidden = {r["source_index"] for r in diagnostic_pool}
    if (len(allowed) != len(pool) or set(allowed) != set(split_manifest["critic_source_indices"])
            or forbidden != set(split_manifest["diagnostic_source_indices"]) or set(allowed) & forbidden):
        raise ValueError("Selection manifest/pools are inconsistent or overlap")
    forbidden_text = {question_key(question_text(r)) for r in diagnostic_pool}
    if any(question_key(question_text(r)) in forbidden_text for r in pool):
        raise ValueError("Critic pool contains question text from the diagnostic pool")
    rows = read_jsonl(args.rollouts_path)
    actual = {r["source_index"] for r in rows}
    if (not actual <= set(allowed) or len(actual) != len(rows)
            or (not args.smoke and actual != set(allowed))):
        raise ValueError("Rollouts must cover exactly the critic pool; --smoke permits a subset")
    if ((not args.smoke and len(rows) != meta["num_questions"])
            or meta["group_size"] < 16):
        raise ValueError("Need at least 16 rollouts per question and matching metadata")
    for row in rows:
        source = allowed[row["source_index"]]
        if question_key(row["question"]) != question_key(question_text(source)) or row["gold_answer"] != gold_answer(source):
            raise ValueError("Question identity/gold answer differs from selected critic pool")
    train_ids, val_ids = question_split(rows, args.validation_fraction, args.split_seed)
    # Disjoint request seed intervals for the original trajectory generator.
    if set(meta["seed"]+r["row_index"] for r in rows) & set(range(diagnostic["seed"], diagnostic["seed"]+len(diagnostic_pool))):
        raise ValueError("Use a separate critic generation seed, e.g. 1000042")
    project_reward_function()
    from transformers import AutoTokenizer
    from treetune.episode_generators.episode_generator_with_reward_function import EpisodeGeneratorWithRewardFunction
    tokenizer = AutoTokenizer.from_pretrained(meta["model_name_or_path"], use_fast=True,
                                             trust_remote_code=meta.get("trust_remote_code", False))
    context = SimpleNamespace(tokenizer=tokenizer)
    examples = {"train": [], "validation": []}
    removed_eos = 0
    noncanonical_tokenizations = 0
    for row in rows:
        if len(row["rollouts"]) != meta["group_size"] or row["group_size"] != meta["group_size"]:
            raise ValueError("Incomplete question rollout group")
        destination = "validation" if row["source_index"] in val_ids else "train"
        for rollout_index, rollout in enumerate(row["rollouts"]):
            reward = score_response(rollout["response_text"], row["gold_answer"], rollout["finish_reason"])
            if not math.isclose(reward, rollout["reward"], abs_tol=1e-12):
                raise ValueError("Stored reward differs from project reward; re-score data first")
            prompt, retokenized_response = EpisodeGeneratorWithRewardFunction._tokenize_trajectory(context,
                dict(query_text=row["prompt_text"], response_text=rollout["response_text"]),
                is_unfinished_response=rollout["finish_reason"] == "length")
            original = list(rollout["response_token_ids"])
            if not retokenized_response or prompt != row["prompt_token_ids"]:
                raise ValueError("Empty response or prompt tokenization drift")
            # A sampled token sequence need not be the tokenizer's canonical
            # segmentation of its decoded text (for example ``p`` + ``ear``
            # versus ``pe`` + ``ar``).  Preserve the actual vLLM token IDs for
            # Critic inputs and use project retokenization only as a text-level
            # consistency check.  The project helper does not append EOS, so a
            # terminal sampled EOS is excluded from the training response.
            sampled_response = original
            if sampled_response and sampled_response[-1] == tokenizer.eos_token_id:
                sampled_response = sampled_response[:-1]
                removed_eos += 1
            decode_kwargs = dict(skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
            sampled_text = tokenizer.decode(sampled_response, **decode_kwargs)
            retokenized_text = tokenizer.decode(retokenized_response, **decode_kwargs)
            if sampled_text != rollout["response_text"] or retokenized_text != rollout["response_text"]:
                raise ValueError(
                    "Project retokenization changes decoded response text; "
                    "regenerate the rollout with a matching tokenizer"
                )
            if retokenized_response != sampled_response:
                noncanonical_tokenizations += 1
            examples[destination].append(dict(source_index=row["source_index"], rollout_index=rollout_index,
                input_ids=prompt+sampled_response, prompt_length=len(prompt), response_length=len(sampled_response),
                reward=reward, finish_reason=rollout["finish_reason"]))
    args.output_dir.mkdir(parents=True)
    for split, data in examples.items():
        with (args.output_dir / (split+".jsonl")).open("w", encoding="utf-8") as handle:
            for example in data:
                handle.write(json.dumps(example, ensure_ascii=False, allow_nan=False)+"\n")
    result = dict(schema="diagnostic_critic_data_v1", status="complete", smoke=args.smoke,
        trajectory_metadata=meta, reward_protocol=REWARD_PROTOCOL,
        split_seed=args.split_seed, validation_fraction=args.validation_fraction,
        train_source_indices=train_ids, validation_source_indices=val_ids,
        diagnostic_source_indices=sorted(forbidden), trajectories_per_question=meta["group_size"],
        train_trajectories=len(examples["train"]), validation_trajectories=len(examples["validation"]),
        removed_terminal_eos_count=removed_eos,
        noncanonical_tokenization_count=noncanonical_tokenizations,
        alignment="sampled vLLM response IDs are preserved; project retokenization is checked at text level; values[P-1:P+T-1] predict s0..s(T-1); terminal value is externally fixed to zero",
        objective="token-weighted MSE to full trajectory project terminal reward; no sigmoid or reward clipping",
        input_hashes={str(p.resolve()): sha256(p) for p in (args.rollouts_path, meta_path, args.diagnostic_metadata_path,
                                                         manifest_path, pool_path, diagnostic_path)},
        data_hashes={split: sha256(args.output_dir / (split+".jsonl")) for split in examples})
    write_json(args.output_dir / "manifest.json", result)
    print(f"Prepared train={len(train_ids)} questions/{len(examples['train'])} trajectories; "
          f"validation={len(val_ids)} questions/{len(examples['validation'])} trajectories")


if __name__ == "__main__":
    main()
