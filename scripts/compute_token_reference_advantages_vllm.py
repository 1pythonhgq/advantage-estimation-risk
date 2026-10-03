"""Estimate all-token reference advantages with two independent MC groups.

Compatible with the repository's vLLM 0.4.0.post1 token-id API. See
scripts/TOKEN_REFERENCE_ADVANTAGES.md for definitions and execution examples.
"""

import argparse
import hashlib
import json
import math
import time
from contextlib import closing
from itertools import zip_longest
from pathlib import Path
from reference_request_scheduler import run_requests

from diagnostic_reference_utils import (
    REWARD_PROTOCOL, completed_response, project_reward_function,
    reference_seed, score_response, summarize_reference,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories-path", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path,
                        help="Default: trajectory filename with .meta.json suffix")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--mc-samples", type=int, default=4,
                        help="K per prefix PER GROUP (at least 2)")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--backend", choices=("engine", "serial"), default="engine",
                        help="engine: continuously batch prefixes with the vLLM 0.4 engine")
    parser.add_argument("--max-pending-requests", type=int, default=16,
                        help="Concurrent prefix requests; each request generates K sequences")
    parser.add_argument("--max-num-seqs", type=int, default=256,
                        help="vLLM scheduler sequence capacity (includes n=K branches)")
    parser.add_argument("--reward-mode", choices=("project", "trajectory"),
                        default="project",
                        help="Project GSM8K reward {-2,0,1}; trajectory is a legacy alias")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)
                    + "\n", encoding="utf-8")


def load_jsonl_records(path, limit=None):
    """Read strict JSONL and report corruption with a useful local excerpt."""
    records = []
    # Do not use str.splitlines(): U+2028/U+2029/U+0085 are legal inside
    # JSON strings and must not be interpreted as JSONL record boundaries.
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                excerpt = line[:160].replace("\n", "\\n")
                raise ValueError(
                    f"Malformed JSONL record at {path}:{line_number}, "
                    f"column {exc.colno}: {exc.msg}. "
                    f"First bytes: {excerpt!r}. Regenerate the trajectory file "
                    "if this is a truncated run."
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected an object at {path}:{line_number}, "
                    f"got {type(record).__name__}."
                )
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    return records


def validate_record(record, tokenizer, metadata):
    """Fail before spending MC compute on an ambiguous/inconsistent trajectory."""
    prompt = record["prompt_token_ids"]
    rollout = record["selected_rollout"]
    tokens = rollout["response_token_ids"]
    if not prompt or not tokens:
        raise ValueError("Empty prompt or selected response")
    if any(type(t) is not int or t < 0 for t in prompt + tokens):
        raise ValueError("Invalid token IDs")
    if len(tokens) > metadata["max_new_tokens"]:
        raise ValueError("Response exceeds the original response token budget")
    if record["rollouts"][record["selected_index"]] != rollout:
        raise ValueError("selected_rollout differs from the recorded selection")
    encoded = tokenizer(record["prompt_text"], add_special_tokens=False)["input_ids"]
    if encoded != prompt:
        raise ValueError("Tokenizer does not reproduce saved prompt IDs")
    # This re-encoding is validation only; all engine inputs use saved IDs.
    if tokenizer.eos_token_id in tokens[:-1]:
        raise ValueError("EOS before the last token; trajectory has nonterminal ambiguity")
    kwargs = dict(skip_special_tokens=True, clean_up_tokenization_spaces=False)
    decoded_prompt = tokenizer.decode(prompt, **kwargs)
    decoded_full = tokenizer.decode(prompt + tokens, **kwargs)
    if decoded_full != decoded_prompt + rollout["response_text"]:
        raise ValueError("Saved response text does not match IDs in prompt context; "
                         "resolve stop/token alignment before reference estimation")
    if metadata["stop_text"] and metadata["stop_text"] in rollout["response_text"]:
        raise ValueError("Saved response contains the excluded stop string")
    expected = score_response(rollout["response_text"], record["gold_answer"],
                              rollout["finish_reason"])
    if not math.isclose(expected, rollout["reward"], abs_tol=1e-12):
        raise ValueError(
            f"Stored reward {rollout['reward']} disagrees with project reward {expected}. "
            "Re-score the saved trajectories with the project rule before MC; "
            "do not mix rewards from different protocols."
        )
    # Guard seed-coordinate bounds before inference.
    reference_seed(metadata["seed"], 1, 0, record["row_index"], len(tokens) - 1)


def iter_record_jobs(record_index, record, sampling_class, metadata, args):
    """Generate jobs lazily; seeds depend on coordinates, never queue order."""
    tokens = record["selected_rollout"]["response_token_ids"]
    for round_index in range(args.rounds):
        for group in (1, 2):
            for prefix_length in range(len(tokens)):
                request_seed = reference_seed(metadata["seed"], group, round_index,
                                              record["row_index"], prefix_length)
                remaining = metadata["max_new_tokens"] - prefix_length
                params = sampling_class(
                    n=args.mc_samples, temperature=metadata["temperature"],
                    top_p=metadata["top_p"], max_tokens=remaining,
                    seed=request_seed,
                    stop=[metadata["stop_text"]] if metadata["stop_text"] else None,
                    include_stop_str_in_output=False,
                )
                yield {"record_index": record_index, "round_index": round_index,
                       "group": group, "prefix_length": prefix_length,
                       "seed": request_seed, "max_tokens": remaining, "params": params,
                       "prompt_token_ids": record["prompt_token_ids"] + tokens[:prefix_length]}


def interleave_jobs(records, sampling_class, metadata, args):
    """Round-robin across trajectories, including unequal response lengths."""
    iterators = [iter_record_jobs(i, record, sampling_class, metadata, args)
                 for i, record in enumerate(records)]
    for row in zip_longest(*iterators):
        for job in row:
            if job is not None:
                yield job


def summarize_record(record, samples, total_tokens, metadata, args):
    rollout = record["selected_rollout"]
    tokens = rollout["response_token_ids"]
    terminal_reward = score_response(rollout["response_text"], record["gold_answer"],
                                     rollout["finish_reason"], args.reward_mode)
    rounds = []
    for round_index in range(args.rounds):
        groups = []
        for group in (1, 2):
            # Explicit coordinate lookup restores token order after asynchronous finishes.
            rewards = [samples[(round_index, group, i)] for i in range(len(tokens))]
            summary = summarize_reference(rewards, terminal_reward)
            seeds = [reference_seed(metadata["seed"], group, round_index, record["row_index"], i)
                     for i in range(len(tokens))]
            summary.update(group=group, seeds=seeds, reward_samples=rewards)
            groups.append(summary)
        rounds.append({"round_index": round_index, "groups": groups})
    return {
        "row_index": record["row_index"], "source_index": record["source_index"],
        "selected_index": record["selected_index"], "prompt_token_ids": record["prompt_token_ids"],
        "response_token_ids": tokens, "response_text": rollout["response_text"],
        "finish_reason": rollout["finish_reason"], "T": len(tokens),
        "stored_trajectory_reward": rollout["reward"], "terminal_reward": terminal_reward,
        "reward_mode": args.reward_mode, "mc_samples_per_group": args.mc_samples,
        "generated_token_count": total_tokens, "rounds": rounds,
    }


def compute_records(records, llm, sampling_class, tokenizer, metadata, args, audit):
    """Yield completed trajectories in completion order, identified by row_index."""
    samples = [{} for _ in records]
    token_counts = [0 for _ in records]
    expected = [len(r["selected_rollout"]["response_token_ids"]) * args.rounds * 2
                for r in records]
    total_requests = sum(expected)
    completed = 0
    generated_tokens = 0
    started = last_report = time.monotonic()
    jobs = interleave_jobs(records, sampling_class, metadata, args)
    with closing(run_requests(llm, jobs, args.max_pending_requests, args.backend)) as results:
        for job, output in results:
            index = job["record_index"]
            record = records[index]
            if len(output.outputs) != args.mc_samples:
                raise ValueError("Unexpected vLLM completion count")
            if list(output.prompt_token_ids) != job["prompt_token_ids"]:
                raise ValueError("Engine altered the fixed token prefix")
            prompt = record["prompt_token_ids"]
            fixed = record["selected_rollout"]["response_token_ids"][:job["prefix_length"]]
            rewards, completions = [], []
            for completion in output.outputs:
                generated = list(completion.token_ids)
                if not generated or len(generated) > job["max_tokens"]:
                    raise ValueError("Empty or over-budget continuation")
                text, effective_finish = completed_response(
                    tokenizer, prompt, fixed + generated, metadata["stop_text"],
                    completion.finish_reason,
                )
                reward = score_response(text, record["gold_answer"], effective_finish, args.reward_mode)
                rewards.append(reward)
                token_counts[index] += len(generated)
                generated_tokens += len(generated)
                completions.append({
                    "token_ids": generated, "engine_text": completion.text,
                    "engine_finish_reason": completion.finish_reason,
                    "effective_finish_reason": effective_finish,
                    "scored_response_text": text, "reward": reward,
                })
            key = (job["round_index"], job["group"], job["prefix_length"])
            if key in samples[index]:
                raise ValueError(f"Duplicate completed reference job: row={record['row_index']}, {key}")
            samples[index][key] = rewards
            audit.write(json.dumps({
                "row_index": record["row_index"], "round_index": job["round_index"],
                "group": job["group"], "prefix_length": job["prefix_length"],
                "seed": job["seed"], "max_tokens": job["max_tokens"],
                "completions": completions,
            }, ensure_ascii=False, allow_nan=False) + "\n")
            completed += 1
            now = time.monotonic()
            if completed % 10 == 0 or now - last_report >= 10 or completed == total_requests:
                audit.flush()
                elapsed = max(now - started, 1e-9)
                print(f"requests={completed}/{total_requests} "
                      f"generated_tokens={generated_tokens} tokens/s={generated_tokens / elapsed:.1f}",
                      flush=True)
                last_report = now
            if len(samples[index]) == expected[index]:
                audit.flush()
                yield summarize_record(record, samples[index], token_counts[index], metadata, args)
                samples[index].clear()
    if completed != total_requests:
        raise RuntimeError("Not all reference prefixes were completed")
    audit.flush()


def compute_record(record, llm, sampling_class, tokenizer, metadata, args, audit):
    """Backwards-compatible serial entry point used by existing callers/tests."""
    from types import SimpleNamespace
    options = SimpleNamespace(**vars(args))
    options.backend = "serial"
    options.max_pending_requests = 1
    return list(compute_records([record], llm, sampling_class, tokenizer, metadata, options, audit))[0]


def main():
    args = parse_args()
    args.reward_mode = "project"  # Normalize the legacy alias in saved metadata.
    if args.mc_samples < 2 or not 1 <= args.rounds <= 1024:
        raise ValueError("Require K >= 2 and rounds in [1, 1024]")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if args.max_pending_requests <= 0 or args.max_num_seqs < args.mc_samples:
        raise ValueError("Require max-pending-requests > 0 and max-num-seqs >= mc-samples")
    meta_path = args.metadata_path or args.trajectories_path.with_suffix(".meta.json")
    raw = args.trajectories_path.read_bytes()
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    required = ("model_name_or_path", "inference_engine", "vllm_version", "temperature",
                "top_p", "max_new_tokens", "stop_text", "seed", "dtype",
                "enable_prefix_caching", "gpu_memory_utilization", "max_model_len")
    for key in required:
        if key not in metadata:
            raise ValueError(f"Trajectory metadata missing {key}; do not guess the protocol")
    if metadata["inference_engine"] != "vllm.LLM":
        raise ValueError("Expected trajectories generated by vllm.LLM")
    if (not math.isfinite(metadata["temperature"]) or metadata["temperature"] <= 0
            or not 0 < metadata["top_p"] <= 1 or metadata["max_new_tokens"] <= 0):
        raise ValueError("Invalid trajectory sampling parameters")
    records = load_jsonl_records(args.trajectories_path, args.limit)
    if not records or len({r["row_index"] for r in records}) != len(records):
        raise ValueError("No records or duplicate row_index values")
    output_meta = args.output_path.with_suffix(".meta.json")
    audit_path = args.output_path.with_suffix(".rollouts.jsonl")
    partial_path = args.output_path.with_suffix(".partial.jsonl")
    targets = [args.output_path, output_meta, audit_path, partial_path]
    if len({p.resolve() for p in targets}) != len(targets):
        raise ValueError("Use an output filename ending in .jsonl")
    for path in targets:
        if path.resolve() in (args.trajectories_path.resolve(), meta_path.resolve()):
            raise ValueError("Output must not overwrite an input")
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {path}; choose a new path or --overwrite")

    # Fail on missing project dependencies before loading model weights.
    project_reward_function()
    import vllm
    from vllm import LLM, SamplingParams

    if vllm.__version__ != metadata["vllm_version"]:
        raise ValueError(f"vLLM version differs: trajectories={metadata['vllm_version']}, "
                         f"current={vllm.__version__}")
    llm_kwargs = dict(model=metadata["model_name_or_path"], dtype=metadata["dtype"],
                      gpu_memory_utilization=metadata["gpu_memory_utilization"],
                      enable_prefix_caching=metadata["enable_prefix_caching"],
                      trust_remote_code=metadata.get("trust_remote_code", False),
                      seed=metadata.get("engine_seed", 0))
    llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if metadata["max_model_len"] is not None:
        llm_kwargs["max_model_len"] = metadata["max_model_len"]
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    context_size = llm.llm_engine.model_config.max_model_len
    for record in records:
        validate_record(record, tokenizer, metadata)
        if len(record["prompt_token_ids"]) + metadata["max_new_tokens"] > context_size:
            raise ValueError("Prompt plus original response budget exceeds context window; "
                             "resolve the trajectory protocol instead of silently shortening MC")
    budget = sum(len(r["selected_rollout"]["response_token_ids"]) for r in records)
    print(f"Questions={len(records)}, prefixes={budget}, "
          f"continuations={budget * 2 * args.mc_samples * args.rounds}", flush=True)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "status": "running", "input_path": str(args.trajectories_path.resolve()),
        "input_sha256": hashlib.sha256(raw).hexdigest(), "trajectory_metadata": metadata,
        "mc_samples_per_group": args.mc_samples, "rounds": args.rounds,
        "backend": args.backend, "max_pending_requests": args.max_pending_requests,
        "max_num_seqs": args.max_num_seqs,
        "record_order": "completion order; join records by row_index, not line number",
        "reward_mode": args.reward_mode, "reward_protocol": REWARD_PROTOCOL, "gamma": 1.0,
        "definition": "z[i] = r[i] + V[i+1] - V[i]; V[T]=0; r[T-1]=R",
        "seed_formula": "group*2**48 + seed + round_index*2**36 + row_index*2**16 + prefix_length",
        "seed_namespace": "Reserve group=1,2 ranges for references, never estimator MC",
        "standard_errors": "sample variance/K; plug-in estimates, not confidence intervals",
        "audit_path": str(audit_path.resolve()), "completed_questions": 0,
    }
    write_json(output_meta, run_meta)
    start = time.monotonic()
    try:
        with partial_path.open("w", encoding="utf-8") as output, \
                audit_path.open("w", encoding="utf-8") as audit:
            with closing(compute_records(records, llm, SamplingParams, tokenizer,
                                         metadata, args, audit)) as results:
                for result in results:
                    output.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                    output.flush()
                    run_meta["completed_questions"] += 1
                    run_meta["generated_token_count"] = (
                        run_meta.get("generated_token_count", 0) + result["generated_token_count"]
                    )
                    write_json(output_meta, run_meta)
        partial_path.replace(args.output_path)
        run_meta.update(status="complete", elapsed_seconds=time.monotonic() - start)
        write_json(output_meta, run_meta)
    except Exception as exc:
        run_meta.update(status="failed", error=str(exc), elapsed_seconds=time.monotonic() - start)
        write_json(output_meta, run_meta)
        raise
    print(f"Saved reference advantages: {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
