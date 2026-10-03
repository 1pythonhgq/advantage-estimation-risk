"""Compute paired-reference metrics for token-level PPO Critic advantages.

PPO produces one advantage for every response token and has no VinePPO/SPO
partition.  Its partition is therefore the singleton-token partition, making
the partition approximation metric A exactly zero by definition.  The risk R
and derived error E are still computed against both independent reference
groups for every recorded reference round.
"""
import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(value)
    if not rows:
        raise ValueError(f"Empty JSONL: {path}")
    return rows


def keyed(rows, field):
    result = {}
    for row in rows:
        key = row.get(field)
        if type(key) is not int or key < 0 or key in result:
            raise ValueError(f"Invalid or duplicate {field}: {key}")
        result[key] = row
    return result


def finite_vector(values, length, name):
    if len(values) != length or any(
        type(value) not in (int, float) or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError(f"Invalid finite vector {name}; expected length={length}")
    return [float(value) for value in values]


def close(left, right):
    return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-10)


def reference_group(group, length, terminal_reward):
    advantages = finite_vector(group["token_advantages"], length, "reference token_advantages")
    values = finite_vector(group["values"], length + 1, "reference values")
    rewards = group["reward_samples"]
    seeds = group["seeds"]
    if len(seeds) != length or len(set(seeds)) != length:
        raise ValueError("Reference group has invalid or duplicate seeds")
    if len(rewards) != length or not rewards or len(rewards[0]) < 2:
        raise ValueError("Reference group has invalid MC samples")
    for index, samples in enumerate(rewards):
        finite_vector(samples, len(samples), "reference reward samples")
        if not close(statistics.mean(samples), values[index]):
            raise ValueError("Reference value does not match reward-sample mean")
        expected = values[index + 1] - values[index]
        if index == length - 1:
            expected += terminal_reward
        if not close(advantages[index], expected):
            raise ValueError("Reference advantage/value identity failed")
    if values[-1] != 0:
        raise ValueError("Reference terminal value must be zero")
    return advantages, set(seeds)


def round_metrics(advantages, reference_one, reference_two):
    length = len(advantages)
    if length == 0:
        raise ValueError("Cannot score an empty response")
    risk = math.fsum(
        (a - z1) * (a - z2)
        for a, z1, z2 in zip(advantages, reference_one, reference_two)
    ) / length
    # PPO is token-wise. Each token is its own block, so both block means are
    # the token values and the segmentation approximation error is zero.
    approximation = 0.0
    return {"R": risk, "A": approximation, "E": risk - approximation}


def bootstrap(values, repetitions, confidence, seed):
    if repetitions <= 0 or len(values) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(repetitions):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(statistics.fmean(sample))
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    low = means[min(len(means) - 1, max(0, int(math.floor(alpha * len(means)))))]
    high_index = min(len(means) - 1, max(0, int(math.ceil((1.0 - alpha) * len(means))) - 1))
    return [low, means[high_index]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references-path", type=Path, required=True)
    parser.add_argument("--ppo-advantages-path", type=Path, required=True)
    parser.add_argument("--project-estimates-path", type=Path,
                        help="project_methods_v2 JSONL used to identify the project token prefix")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--token-scope", choices=("full", "project"), default="full")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    if not 0.0 < args.confidence < 1.0 or args.bootstrap < 0:
        raise ValueError("Invalid confidence or bootstrap count")
    if args.token_scope == "project" and args.project_estimates_path is None:
        raise ValueError("--project-estimates-path is required with --token-scope project")

    ref_meta_path = args.references_path.with_suffix(".meta.json")
    ppo_meta_path = args.ppo_advantages_path.with_suffix(".meta.json")
    ref_meta = json.loads(ref_meta_path.read_text(encoding="utf-8"))
    ppo_meta = json.loads(ppo_meta_path.read_text(encoding="utf-8"))
    if ref_meta.get("status") != "complete" or ppo_meta.get("status") != "complete":
        raise ValueError("Reference and PPO inputs must both have status=complete")
    if ppo_meta.get("schema") != "ppo_critic_advantages_v1":
        raise ValueError("The PPO input is not ppo_critic_advantages_v1")
    if ppo_meta.get("whiten_advantages") is not False:
        raise ValueError("PPO metrics require unwhitened token advantages")
    ref_hash = ref_meta.get("input_sha256")
    ppo_hash = ppo_meta.get("input_sha256")
    if not ref_hash or ref_hash != ppo_hash:
        raise ValueError("Reference and PPO advantages were not computed from the same trajectory file")
    if ref_meta.get("reward_protocol") != ppo_meta.get("reward_protocol"):
        raise ValueError("Reference and PPO reward protocols differ")

    references = keyed(read_jsonl(args.references_path), "row_index")
    ppo_rows = keyed(read_jsonl(args.ppo_advantages_path), "row_index")
    project_rows = None
    tokenizer_eos_id = None
    if args.project_estimates_path is not None:
        project_meta_path = args.project_estimates_path.with_suffix(".meta.json")
        project_meta = json.loads(project_meta_path.read_text(encoding="utf-8"))
        if project_meta.get("status") != "complete" or project_meta.get("schema") != "project_methods_v2":
            raise ValueError("Project estimates must be a complete project_methods_v2 run")
        if project_meta.get("trajectory_sha256") != ppo_hash:
            raise ValueError("Project estimates and PPO advantages use different trajectory files")
        project_rows = keyed(read_jsonl(args.project_estimates_path), "row_index")
        if args.token_scope == "project":
            model_path = ppo_meta.get("trajectory_metadata", {}).get("model_name_or_path")
            if not model_path:
                raise ValueError("PPO metadata does not record the tokenizer/model path")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_fast=True,
                trust_remote_code=ppo_meta.get("trajectory_metadata", {}).get("trust_remote_code", False),
            )
            tokenizer_eos_id = tokenizer.eos_token_id
            if tokenizer_eos_id is None:
                raise ValueError("Tokenizer has no eos_token_id")
    if references.keys() != ppo_rows.keys():
        raise ValueError("Reference and PPO question sets differ")
    if project_rows is not None and project_rows.keys() != references.keys():
        raise ValueError("Project estimates and PPO question sets differ")

    per_question = []
    all_reference_seeds = set()
    for row_index in sorted(references):
        reference = references[row_index]
        ppo = ppo_rows[row_index]
        if ppo.get("selected") is not True:
            raise ValueError(f"row={row_index}: PPO input is not the selected rollout")
        for field in ("source_index", "response_text", "finish_reason", "terminal_reward"):
            if reference.get(field) != ppo.get(field):
                raise ValueError(f"row={row_index}: identity mismatch: {field}")
        response_ids = reference.get("response_token_ids")
        if response_ids != ppo.get("response_token_ids"):
            raise ValueError(f"row={row_index}: response token IDs differ")
        full_length = len(response_ids)
        full_advantages = finite_vector(
            ppo.get("token_advantages", []), full_length, "PPO token_advantages"
        )
        evaluation_indices = list(range(full_length))
        if args.token_scope == "project":
            project = project_rows[row_index]
            if project.get("original_response_token_ids") != response_ids:
                raise ValueError(f"row={row_index}: project/original response IDs differ")
            evaluation_indices = project.get("reference_token_indices")
            if evaluation_indices != list(range(len(evaluation_indices))):
                raise ValueError(f"row={row_index}: project token mapping is not a prefix")
            if project.get("excluded_reference_token_indices") != list(range(len(evaluation_indices), full_length)):
                raise ValueError(f"row={row_index}: invalid project excluded-token mapping")
            if len(evaluation_indices) == 0 or len(evaluation_indices) > full_length:
                raise ValueError(f"row={row_index}: invalid project token count")
            excluded = list(range(len(evaluation_indices), full_length))
            if len(excluded) > 1:
                raise ValueError(f"row={row_index}: project mapping excludes more than one token")
            if excluded and response_ids[excluded[0]] != tokenizer_eos_id:
                raise ValueError(
                    f"row={row_index}: excluded token {response_ids[excluded[0]]} "
                    f"is not tokenizer EOS {tokenizer_eos_id}"
                )
        length = len(evaluation_indices)
        rounds = reference.get("rounds")
        if not isinstance(rounds, list) or not rounds:
            raise ValueError(f"row={row_index}: missing reference rounds")
        round_results = []
        for reference_round in rounds:
            groups = {group.get("group"): group for group in reference_round.get("groups", [])}
            if set(groups) != {1, 2}:
                raise ValueError(f"row={row_index}: expected two reference groups")
            z1_full, seeds_one = reference_group(groups[1], full_length, reference["terminal_reward"])
            z2_full, seeds_two = reference_group(groups[2], full_length, reference["terminal_reward"])
            z1 = [z1_full[index] for index in evaluation_indices]
            z2 = [z2_full[index] for index in evaluation_indices]
            advantages = [full_advantages[index] for index in evaluation_indices]
            if seeds_one & seeds_two or all_reference_seeds & (seeds_one | seeds_two):
                raise ValueError("Reference request seeds are not independent")
            all_reference_seeds.update(seeds_one | seeds_two)
            round_results.append({
                "round_index": reference_round["round_index"],
                "metrics": round_metrics(advantages, z1, z2),
                "segment_count": length,
            })
        averages = {
            metric: statistics.fmean(item["metrics"][metric] for item in round_results)
            for metric in ("R", "A", "E")
        }
        per_question.append({
            "row_index": row_index,
            "source_index": reference["source_index"],
            "token_count": length,
            "round_count": len(round_results),
            "rounds": round_results,
            "metrics": averages,
        })

    summary = {"method": "ppo", "aggregation": "equal question weights after averaging rounds"}
    for metric in ("R", "A", "E"):
        question_values = [row["metrics"][metric] for row in per_question]
        token_weighted_numerator = math.fsum(
            item["metrics"][metric] * row["token_count"]
            for row in per_question for item in row["rounds"]
        )
        token_weighted_denominator = sum(
            row["token_count"] for row in per_question for _ in row["rounds"]
        )
        summary[metric] = {
            "question_mean": statistics.fmean(question_values),
            "token_weighted_mean": token_weighted_numerator / token_weighted_denominator,
            "confidence_interval": bootstrap(
                question_values, args.bootstrap, args.confidence,
                args.seed + {"R": 11, "A": 23, "E": 37}[metric],
            ),
        }

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "per_question.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_question:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps({
            "schema": "ppo_diagnostic_metrics_v1",
            "status": "complete",
            "method": "ppo",
            "metric_definition": {
                "R": "mean((ppo_advantage-reference_1)*(ppo_advantage-reference_2))",
                "A": "singleton-token partition approximation error; zero by definition",
                "E": "R-A",
            },
            "reference_path": str(args.references_path.resolve()),
            "ppo_advantages_path": str(args.ppo_advantages_path.resolve()),
            "project_estimates_path": (str(args.project_estimates_path.resolve())
                                        if args.project_estimates_path else None),
            "token_scope": args.token_scope,
            "trajectory_sha256": ppo_hash,
            "question_count": len(per_question),
            "bootstrap": args.bootstrap,
            "confidence": args.confidence,
            "metrics": summary,
        }, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "metric", "question_mean", "ci_low", "ci_high", "token_weighted_mean"])
        for metric in ("R", "A", "E"):
            entry = summary[metric]
            interval = entry["confidence_interval"] or [None, None]
            writer.writerow(["ppo", metric, entry["question_mean"], *interval, entry["token_weighted_mean"]])
    print(f"Saved PPO diagnostic metrics to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
