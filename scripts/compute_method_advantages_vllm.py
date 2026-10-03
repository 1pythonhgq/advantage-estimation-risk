"""Run the project's GRPO/VinePPO/SPO-int5 estimators, without SPO masking."""

import argparse
import hashlib
import json
import math
import time
from contextlib import closing
from itertools import zip_longest
from pathlib import Path

from compute_token_reference_advantages_vllm import load_jsonl_records, validate_record, write_json
from diagnostic_reference_utils import REWARD_PROTOCOL, project_reward_function, score_response
from project_method_adapter import (SCHEMA, ProjectMethods, estimator_seed, load_project_protocol,
                                    source_fingerprint, validate_sampling_metadata)
from reference_request_scheduler import run_requests


def validate_partitions(record, partition, methods):
    if partition.get("schema") != SCHEMA:
        raise ValueError("Old partition format; rerun prepare_method_partitions.py")
    if methods.prepare(record, partition["actor_token_logprobs"]) != partition:
        raise ValueError(f"Prepared data differs from project logic: row={record['row_index']}")


def validate_group(record):
    rollouts = record["rollouts"]
    if len(rollouts) != record["group_size"] or len(rollouts) < 2:
        raise ValueError("Invalid original GRPO group size")
    if not 0 <= record["selected_index"] < len(rollouts):
        raise ValueError("Invalid selected_index")
    rewards = [score_response(r["response_text"], record["gold_answer"], r["finish_reason"])
               for r in rollouts]
    for saved in ([r["reward"] for r in rollouts], record["group_rewards"]):
        if len(saved) != len(rewards) or any(
                not math.isclose(a, b, abs_tol=1e-12) for a, b in zip(saved, rewards)):
            raise ValueError("Original group rewards disagree with project reward; re-score first")


def record_requests(index, record, partition, args):
    for round_index in range(args.rounds):
        for method in ("vineppo", "spo_int5"):
            for request in partition[method]["requests"]:
                yield dict(record_index=index, round_index=round_index, method=method,
                           count=args.mc_samples, **request)
        if round_index > 0:
            yield dict(record_index=index, round_index=round_index, method="grpo", value_index=0,
                       count=record["group_size"]-1, query_text=record["prompt_text"])


def build_jobs(records, partitions, sampling_class, metadata, args, methods):
    streams = [record_requests(i, r, p, args)
               for i, (r, p) in enumerate(zip(records, partitions))]
    unique = {}
    for jobs in zip_longest(*streams):
        for job in jobs:
            if job is None:
                continue
            # Match project request deduplication while keeping methods/rounds independent.
            key = (job["method"], job["round_index"], job["query_text"],
                   job["record_index"] if job["method"] == "grpo" else None)
            if key not in unique:
                unique[key] = {**job, "targets": []}
            unique[key]["targets"].append((job["record_index"], job["value_index"]))
    for job in unique.values():
        row, value_index = min((records[i]["row_index"], v) for i, v in job["targets"])
        job["seed"] = estimator_seed(metadata["seed"], job["method"], job["round_index"], row, value_index)
        # Project requests are TEXT. Use the same tokenizer default as the vLLM text API.
        job["prompt_token_ids"] = list(methods.tokenizer.encode(job["query_text"]))
        job["max_tokens"] = methods.max_tokens(job["query_text"], metadata["max_new_tokens"])
        if job["max_tokens"] <= 0:
            raise ValueError("Project MC request exhausts configured context budget")
        job["params"] = sampling_class(n=job["count"], temperature=metadata["temperature"], top_p=metadata["top_p"],
            max_tokens=job["max_tokens"], seed=job["seed"],
            stop=[metadata["stop_text"]] if metadata["stop_text"] else None, include_stop_str_in_output=False)
        yield job


def summarize_record(record, partition, samples, token_count, metadata, args, methods):
    rollout = record["selected_rollout"]
    rounds = []
    for round_index in range(args.rounds):
        rewards = list(record["group_rewards"]) if round_index == 0 else list(samples[(round_index, "grpo", 0)]["reward_samples"])
        if round_index > 0:
            rewards.insert(record["selected_index"], rollout["reward"])
        grpo = methods.grpo_estimate(rewards, record["selected_index"], len(partition["trajectory"]["response_token_ids"]))
        grpo["group_source"] = "original_trajectory_group" if round_index == 0 else "fixed_selected_plus_fresh_others"
        grpo["seed"] = None if round_index == 0 else estimator_seed(
            metadata["seed"], "grpo", round_index, record["row_index"], 0)
        result = dict(round_index=round_index, grpo=grpo)
        for method in ("vineppo", "spo_int5"):
            stats = {r["value_index"]: samples[(round_index, method, r["value_index"])]
                     for r in partition[method]["requests"]}
            estimate = methods.estimate(partition, method, stats)
            estimate["mc_requests"] = [dict(value_index=i, **stat) for i, stat in sorted(stats.items())]
            result[method] = estimate
        for method in ("grpo", "vineppo", "spo_int5"):
            advantages = result[method]["token_advantages"]
            if (len(advantages) != len(partition["trajectory"]["response_token_ids"])
                    or not all(math.isfinite(a) for a in advantages)):
                raise ValueError(f"Invalid project advantage output: {method}")
        rounds.append(result)
    return dict(row_index=record["row_index"], source_index=record["source_index"],
                selected_index=record["selected_index"], prompt_token_ids=record["prompt_token_ids"],
                schema=SCHEMA, response_token_ids=partition["trajectory"]["response_token_ids"],
                original_response_token_ids=rollout["response_token_ids"],
                reference_token_indices=partition["reference_token_indices"],
                excluded_reference_token_indices=partition["excluded_reference_token_indices"],
                response_text=rollout["response_text"],
                finish_reason=rollout["finish_reason"], terminal_reward=rollout["reward"],
                T=len(partition["trajectory"]["response_token_ids"]), mc_samples_per_request=args.mc_samples,
                generated_token_count=token_count, rounds=rounds)


def compute_records(records, partitions, llm, sampling_class, methods, metadata, args, audit):
    samples = [{} for _ in records]
    token_counts = [0 for _ in records]
    expected = [args.rounds * sum(len(p[m]["requests"]) for m in ("vineppo", "spo_int5"))
                + args.rounds - 1 for p in partitions]
    completed, generated_tokens = 0, 0
    started = last_report = time.monotonic()
    jobs = list(build_jobs(records, partitions, sampling_class, metadata, args, methods))
    total_requests = len(jobs)
    for job in jobs:
        if len(job["prompt_token_ids"]) + job["max_tokens"] > llm.llm_engine.model_config.max_model_len:
            raise ValueError("Project request exceeds actual vLLM context; do not silently truncate")
    with closing(run_requests(llm, jobs, args.max_pending_requests, args.backend)) as results:
        for job, output in results:
            if len(output.outputs) != job["count"] or list(output.prompt_token_ids) != job["prompt_token_ids"]:
                raise ValueError("Engine changed completion count or fixed prefix")
            completions = []
            request_tokens = 0
            for completion in output.outputs:
                generated = list(completion.token_ids)
                if len(generated) > job["max_tokens"]:
                    raise ValueError("Over-budget continuation")
                request_tokens += len(generated)
                completions.append(dict(token_ids=generated, engine_text=completion.text,
                                        engine_finish_reason=completion.finish_reason))
            generated_tokens += request_tokens
            targets = []
            for position, (index, value_index) in enumerate(job["targets"]):
                record = records[index]
                stat, children = methods.score_completions(record, job["method"], job["query_text"], output.outputs)
                key = (job["round_index"], job["method"], value_index)
                if key in samples[index]:
                    raise ValueError(f"Duplicate project value destination: {key}")
                samples[index][key] = dict(**stat, seed=job["seed"])
                # Charge each deduplicated request once, so total generation counts stay exact.
                if position == 0:
                    token_counts[index] += request_tokens
                targets.append(dict(row_index=record["row_index"], value_index=value_index,
                                    statistics=stat, scored_completions=children))
            audit_record = {k: job[k] for k in ("round_index", "method", "query_text", "prompt_token_ids", "seed", "max_tokens")}
            audit_record.update(targets=targets, completions=completions)
            audit.write(json.dumps(audit_record, ensure_ascii=False, allow_nan=False) + "\n")
            completed += 1
            now = time.monotonic()
            if completed % 10 == 0 or now - last_report >= 10 or completed == total_requests:
                audit.flush()
                print(f"requests={completed}/{total_requests} generated_tokens={generated_tokens} "
                      f"tokens/s={generated_tokens/max(now-started, 1e-9):.1f}", flush=True)
                last_report = now
            for index in sorted({i for i, _ in job["targets"]}):
                if len(samples[index]) == expected[index]:
                    audit.flush()
                    yield summarize_record(records[index], partitions[index], samples[index], token_counts[index], metadata, args, methods)
                    samples[index].clear()
    if completed != total_requests:
        raise RuntimeError("Not all estimator requests completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories-path", type=Path, required=True)
    parser.add_argument("--partitions-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--mc-samples", type=int, default=9)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--backend", choices=("engine", "serial"), default="engine")
    parser.add_argument("--max-pending-requests", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    args = parser.parse_args()
    if args.mc_samples < 2 or not 1 <= args.rounds <= 1024 or args.max_pending_requests <= 0:
        raise ValueError("Require K>=2, rounds in [1,1024], positive pending request limit")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    metadata_path = args.trajectories_path.with_suffix(".meta.json")
    partition_meta_path = args.partitions_path.with_suffix(".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    partition_meta = json.loads(partition_meta_path.read_text(encoding="utf-8"))
    if partition_meta.get("schema") != SCHEMA:
        raise ValueError("Old partition protocol; regenerate partitions with the updated preparation script")
    protocol = load_project_protocol(metadata["seed"])
    validate_sampling_metadata(metadata, protocol)
    if (partition_meta["project_protocol"] != protocol
            or partition_meta["source_fingerprint"] != source_fingerprint()):
        raise ValueError("Project code/config changed since preparation; regenerate partitions")
    if args.mc_samples != protocol["mc_samples"]:
        raise ValueError(f"Project MC budget is {protocol['mc_samples']}; keep --mc-samples consistent")
    digest = hashlib.sha256(args.trajectories_path.read_bytes()).hexdigest()
    if (partition_meta["status"] != "complete" or partition_meta["input_sha256"] != digest
            or partition_meta["trajectory_metadata"] != metadata):
        raise ValueError("Partitions were not completed against these exact trajectories/metadata")
    required = ("model_name_or_path", "inference_engine", "vllm_version", "temperature", "top_p",
                "max_new_tokens", "stop_text", "seed", "dtype", "enable_prefix_caching",
                "gpu_memory_utilization", "max_model_len")
    if any(k not in metadata for k in required) or metadata["inference_engine"] != "vllm.LLM":
        raise ValueError("Missing or incompatible trajectory engine metadata")
    if (not math.isfinite(metadata["temperature"]) or metadata["temperature"] <= 0
            or not 0 < metadata["top_p"] <= 1 or metadata["max_new_tokens"] <= 0):
        raise ValueError("Invalid sampling metadata")
    records = load_jsonl_records(args.trajectories_path, args.limit)
    prepared = load_jsonl_records(args.partitions_path)
    by_row = {p["row_index"]: p for p in prepared}
    if (not records or len({r["row_index"] for r in records}) != len(records)
            or len(by_row) != len(prepared)):
        raise ValueError("Empty or duplicate input rows")
    partitions = [by_row[r["row_index"]] for r in records]
    for record, partition in zip(records, partitions):
        if partition.get("schema") != SCHEMA:
            raise ValueError("Old partition row; regenerate partitions")
        validate_group(record)
        estimator_seed(metadata["seed"], "spo_int5", args.rounds-1, record["row_index"],
                       len(record["selected_rollout"]["response_token_ids"])-1)
    if args.max_num_seqs < max([args.mc_samples] + [r["group_size"]-1 for r in records]):
        raise ValueError("max-num-seqs must accommodate at least one request's samples")
    output_meta = args.output_path.with_suffix(".meta.json")
    partial = args.output_path.with_suffix(".partial.jsonl")
    audit_path = args.output_path.with_suffix(".rollouts.jsonl")
    if args.output_path.suffix != ".jsonl":
        raise ValueError("Use .jsonl output")
    inputs = {p.resolve() for p in (args.trajectories_path, metadata_path, args.partitions_path, partition_meta_path)}
    for path in (args.output_path, output_meta, partial, audit_path):
        if path.exists() or path.resolve() in inputs:
            raise FileExistsError(f"Choose a fresh output path: {path}")
    project_reward_function()
    import vllm
    from vllm import LLM, SamplingParams
    if vllm.__version__ != metadata["vllm_version"]:
        raise ValueError("vLLM version differs from trajectory generation")
    options = dict(model=metadata["model_name_or_path"], dtype=metadata["dtype"],
                   gpu_memory_utilization=metadata["gpu_memory_utilization"],
                   enable_prefix_caching=metadata["enable_prefix_caching"],
                   trust_remote_code=metadata.get("trust_remote_code", False),
                   seed=metadata.get("engine_seed", 0), max_num_seqs=args.max_num_seqs)
    if metadata["max_model_len"] is not None:
        options["max_model_len"] = metadata["max_model_len"]
    llm = LLM(**options)
    tokenizer = llm.get_tokenizer()
    methods = ProjectMethods(tokenizer, protocol)
    for record, partition in zip(records, partitions):
        validate_record(record, tokenizer, metadata)
        validate_partitions(record, partition, methods)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(status="running", trajectory_sha256=digest, trajectory_metadata=metadata,
                    partitions_sha256=hashlib.sha256(args.partitions_path.read_bytes()).hexdigest(),
                    partition_metadata=partition_meta, reward_protocol=REWARD_PROTOCOL,
                    schema=SCHEMA, project_protocol=protocol, mc_samples_per_request=args.mc_samples, rounds=args.rounds,
                    backend=args.backend, max_pending_requests=args.max_pending_requests,
                    max_num_seqs=args.max_num_seqs, gamma=1.0, batch_whitening=False, probability_mask=False,
                    variant="project_estimation_without_spo_probability_mask",
                    mc_prefix_protocol="project text queries re-encoded with vLLM tokenizer defaults",
                    mc_length_protocol="project EfficientIIDExpander._compute_max_tokens",
                    reference_compatibility="Token mapping is not target equivalence: verify horizon and terminal/EOS definition before comparison",
                    seed_namespaces=dict(reference=[1, 2], vineppo=3, spo_int5=4, grpo=5),
                    seed_formula="namespace*2**48 + seed + round*2**36 + canonical_row_index*2**16 + value_index",
                    record_order="completion order; join by row_index and reference_token_indices",
                    audit_path=str(audit_path.resolve()), completed_questions=0)
    write_json(output_meta, manifest)
    started = time.monotonic()
    try:
        with partial.open("w", encoding="utf-8") as output, audit_path.open("w", encoding="utf-8") as audit:
            with closing(compute_records(records, partitions, llm, SamplingParams, methods, metadata, args, audit)) as results:
                for result in results:
                    output.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                    output.flush()
                    manifest["completed_questions"] += 1
                    manifest["generated_token_count"] = manifest.get("generated_token_count", 0) + result["generated_token_count"]
                    write_json(output_meta, manifest)
        partial.replace(args.output_path)
        manifest.update(status="complete", elapsed_seconds=time.monotonic()-started)
        write_json(output_meta, manifest)
    except Exception as exc:
        manifest.update(status="failed", error=str(exc), elapsed_seconds=time.monotonic()-started)
        write_json(output_meta, manifest)
        raise
    print(f"Saved method advantages: {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
