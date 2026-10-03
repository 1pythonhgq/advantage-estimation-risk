"""CPU-only paired-reference risk R and partition approximation error A."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics

METHODS = ("grpo", "grpo_unstandardized", "vineppo", "spo_int5")
METRICS = ("R", "A", "E")


def read_jsonl(path):
    # File iteration preserves GSM8K's legal U+2028 within JSON strings.
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"Invalid JSONL: {path}:{number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object: {path}:{number}")
            rows.append(value)
    return rows


def keyed(items, field):
    result = {}
    for item in items:
        key = item[field]
        if type(key) is not int or key < 0 or key in result:
            raise ValueError(f"Invalid or duplicate {field}: {key}")
        result[key] = item
    if not result:
        raise ValueError(f"Empty {field} collection")
    return result


def vector(values, length, name):
    if len(values) != length or any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
        raise ValueError(f"Invalid finite vector: {name}, expected length={length}")
    return values


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-10)


def reference_group(group, length, terminal_reward):
    z = vector(group["token_advantages"], length, "reference advantages")
    values = vector(group["values"], length+1, "reference values")
    seeds = group["seeds"]
    if len(seeds) != length or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != length:
        raise ValueError("Missing, duplicate or invalid reference request seeds")
    rewards = group["reward_samples"]
    if len(rewards) != length or len({len(s) for s in rewards}) != 1 or len(rewards[0]) < 2:
        raise ValueError("Invalid reference MC budgets")
    for i, samples in enumerate(rewards):
        vector(samples, len(samples), "reference rewards")
        if not close(statistics.mean(samples), values[i]):
            raise ValueError("Reference values do not match reward samples")
    if values[-1] != 0:
        raise ValueError("Reference terminal value must be zero")
    for i, advantage in enumerate(z):
        expected = values[i+1]-values[i] + (terminal_reward if i == length-1 else 0)
        if not close(advantage, expected):
            raise ValueError("Reference advantage/value identity failed")
    return z, set(seeds), len(rewards[0])


def blocks_for(method, estimate, length):
    if method.startswith("grpo"):
        return [list(range(length))]
    if method == "vineppo":
        ids = estimate["token_step_indices"]
        chars = estimate["character_boundaries"]
        if (len(chars) < 2 or chars[0] != 0 or any(type(x) is not int for x in chars)
                or any(a > b for a, b in zip(chars, chars[1:]))):
            raise ValueError("Invalid original VinePPO character boundaries")
        if (len(ids) != length or any(type(x) not in (int, float) or not math.isfinite(x)
                or x != int(x) or not 0 <= x < len(chars)-1 for x in ids)
                or ids != sorted(ids)):
            raise ValueError("Invalid project token-to-step assignment")
        # Empty character steps remain in the source data; they contribute no tokens.
        return [list(indices) for _, indices in itertools.groupby(range(length), key=lambda i: ids[i])]
    cuts = estimate["cutpoints"]
    if (any(type(c) is not int or not -1 <= c < length-1 for c in cuts)
            or any(a >= b for a, b in zip(cuts, cuts[1:]))):
        raise ValueError("Invalid project SPO cutpoints")
    # cutpoint c is the last token of a segment, not the first of the next.
    ends = [c+1 for c in cuts if c >= 0] + [length]
    starts = [0] + ends[:-1]
    return [list(range(a, b)) for a, b in zip(starts, ends)]


def round_metrics(advantages, z1, z2, blocks):
    length = len(z1)
    vector(advantages, length, "method advantages")
    vector(z2, length, "second reference")
    if not length or [i for block in blocks for i in block] != list(range(length)):
        raise ValueError("Partition must cover every evaluated token exactly once in order")
    risk = math.fsum((a-x)*(a-y) for a, x, y in zip(advantages, z1, z2))/length
    approximation_terms, projected_terms = [], []
    for block in blocks:
        if not block:
            raise ValueError("Empty token block")
        if any(not close(advantages[i], advantages[block[0]]) for i in block):
            raise ValueError("Method advantages are not constant inside their project segment")
        mean1 = statistics.mean(z1[i] for i in block)
        mean2 = statistics.mean(z2[i] for i in block)
        approximation_terms.extend((z1[i]-mean1)*(z2[i]-mean2) for i in block)
        projected_terms.append(len(block)*(advantages[block[0]]-mean1)*(advantages[block[0]]-mean2))
    approximation = math.fsum(approximation_terms)/length
    error = risk-approximation
    if not close(error, math.fsum(projected_terms)/length):
        raise ValueError("R=A+E projection identity failed")
    return dict(R=risk, A=approximation, E=error)


def assess_metadata(reference, estimates, comparison):
    for meta in (reference, estimates):
        if meta.get("status") != "complete":
            raise ValueError("Both input runs must have status=complete")
    if estimates.get("schema") != "project_methods_v2":
        raise ValueError("Use project_methods_v2 method estimates, not the previous partition-only variant")
    if not reference.get("input_sha256") or reference["input_sha256"] != estimates.get("trajectory_sha256"):
        raise ValueError("Reference and methods were not computed from the same trajectory file")
    for key in ("trajectory_metadata", "reward_protocol", "gamma", "rounds"):
        if key not in reference or key not in estimates or reference[key] != estimates[key]:
            raise ValueError(f"Reference/method metadata mismatch: {key}")
    if estimates.get("probability_mask") is not False or estimates.get("batch_whitening") is not False:
        raise ValueError("Expected unmasked, pre-whitening method advantages")
    # Only explicit matching target descriptors establish strict protocol compatibility.
    ref_target, est_target = reference.get("target_protocol"), estimates.get("target_protocol")
    verified_same = bool(ref_target) and ref_target == est_target
    if comparison == "strict" and not verified_same:
        raise ValueError("Identical target protocols are not established. Existing reference uses fixed token prefixes "
                         "and remaining response budget; project methods use text queries and project MC budget. "
                         "Use --comparison reference-target to explicitly evaluate methods against that existing "
                         "reference target, or regenerate compatible inputs. Do not relabel metadata to bypass this check.")
    return dict(comparison=comparison, identical_target_protocol_verified=verified_same,
                interpretation=("risk against the supplied reference target; estimator error may include differences "
                                "in rollout horizon, prefix encoding and terminal handling" if not verified_same
                                else "risk against an explicitly matching target protocol"),
                reference_target_protocol=ref_target,
                estimator_prefix_protocol=estimates.get("mc_prefix_protocol"),
                estimator_length_protocol=estimates.get("mc_length_protocol"))


def calculate_rows(references, estimates, token_scope):
    refs, methods = keyed(references, "row_index"), keyed(estimates, "row_index")
    if refs.keys() != methods.keys():
        raise ValueError("Question sets differ; no implicit intersection or dropping rows is allowed")
    if len({r["source_index"] for r in references}) != len(references):
        raise ValueError("Expected one selected trajectory per question; duplicate source_index")
    all_ref_seeds = set()
    all_est_seeds = set()
    estimator_owners = {}
    result = []
    round_sets = []
    for row in sorted(refs):
        ref, est = refs[row], methods[row]
        for key in ("source_index", "selected_index", "prompt_token_ids", "response_text", "finish_reason", "terminal_reward"):
            if ref[key] != est[key]:
                raise ValueError(f"row={row}: identity mismatch: {key}")
        if not math.isfinite(ref["terminal_reward"]):
            raise ValueError("Invalid terminal reward")
        original, tokens = ref["response_token_ids"], est["response_token_ids"]
        if ref["T"] != len(original) or est["T"] != len(tokens) or not tokens:
            raise ValueError("Invalid T/token IDs")
        if original != est["original_response_token_ids"]:
            raise ValueError(f"row={row}: original response IDs differ")
        mapping = est["reference_token_indices"]
        if (len(tokens) > len(original) or mapping != list(range(len(tokens)))
                or [original[i] for i in mapping] != tokens):
            raise ValueError("Unsupported token mapping; expected matching project token prefix")
        excluded = list(range(len(tokens), len(original)))
        if excluded != est["excluded_reference_token_indices"] or len(excluded) > 1:
            raise ValueError("Invalid excluded token mapping")
        if token_scope == "full" and excluded:
            raise ValueError(f"row={row}: method excludes a terminal reference token. Use --token-scope project "
                             "to explicitly evaluate only project tokens; no EOS advantage will be invented.")
        ref_rounds, est_rounds = keyed(ref["rounds"], "round_index"), keyed(est["rounds"], "round_index")
        if ref_rounds.keys() != est_rounds.keys() or sorted(ref_rounds) != list(range(len(ref_rounds))):
            raise ValueError("Rounds must match exactly and be contiguous from zero")
        round_sets.append(tuple(sorted(ref_rounds)))
        per_round, previous_blocks = [], {}
        for rnd in sorted(ref_rounds):
            groups = keyed(ref_rounds[rnd]["groups"], "group")
            if set(groups) != {1, 2}:
                raise ValueError("Exactly two independent reference groups are required")
            z = []
            budgets = []
            for group in (1, 2):
                values, seeds, budget = reference_group(groups[group], len(original), ref["terminal_reward"])
                if seeds & all_ref_seeds:
                    raise ValueError("Reference requests reuse seeds across groups, rounds or questions")
                all_ref_seeds.update(seeds)
                z.append([values[i] for i in mapping])
                budgets.append(budget)
            if budgets[0] != budgets[1] or budgets[0] != ref["mc_samples_per_group"]:
                raise ValueError("Reference groups must have the recorded equal MC budget")
            entry = dict(round_index=rnd, methods={})
            for name in METHODS:
                base = "grpo" if name == "grpo_unstandardized" else name
                method = est_rounds[rnd][base]
                field = "token_advantages_unstandardized" if name == "grpo_unstandardized" else "token_advantages"
                blocks = blocks_for(name, method, len(tokens))
                if name in previous_blocks and previous_blocks[name] != blocks:
                    raise ValueError("Partition changed across repeated rounds on a fixed trajectory")
                previous_blocks[name] = blocks
                metrics = round_metrics(method[field], z[0], z[1], blocks)
                entry["methods"][name] = dict(**metrics, nonempty_token_segments=len(blocks))
                if name == "grpo_unstandardized":
                    continue
                seeds = ([method["seed"]] if name == "grpo" and rnd > 0 else
                         [] if name == "grpo" else [req["seed"] for req in method["mc_requests"]])
                for seed in seeds:
                    if type(seed) is not int or seed < 0:
                        raise ValueError("Invalid estimator seed")
                    # Project query deduplication may share a request within one method/round.
                    owner = (name, rnd)
                    if seed in estimator_owners and estimator_owners[seed] != owner:
                        raise ValueError("Estimator streams overlap across methods or rounds")
                    estimator_owners[seed] = owner
                    all_est_seeds.add(seed)
            per_round.append(entry)
        averages = {name: {metric: statistics.mean(r["methods"][name][metric] for r in per_round)
                           for metric in METRICS} for name in METHODS}
        result.append(dict(row_index=row, source_index=ref["source_index"], selected_index=ref["selected_index"],
                           reference_T=len(original), evaluated_T=len(tokens), round_count=len(per_round),
                           reference_token_indices=mapping, excluded_reference_token_indices=excluded,
                           rounds=per_round, methods=averages))
    if len(set(round_sets)) != 1:
        raise ValueError("Different numbers of rounds across questions")
    if all_ref_seeds & all_est_seeds:
        raise ValueError("Estimator seeds overlap reference seeds; paired-risk independence is violated")
    return result, dict(reference_unique_request_seeds=len(all_ref_seeds),
                        estimator_unique_request_seeds=len(all_est_seeds), no_recorded_seed_overlap=True,
                        limitation="Seed checks cannot prove independent RNG implementations or unchanged model weights")


def percentile(values, q):
    values = sorted(values)
    position = (len(values)-1)*q
    left = int(position)
    right = min(left+1, len(values)-1)
    return values[left] + (values[right]-values[left])*(position-left)


def aggregate(rows, repetitions=1000, seed=42, confidence=.95):
    if repetitions < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap configuration")
    n = len(rows)
    features = {(name, metric): [r["methods"][name][metric] for r in rows]
                for name in METHODS for metric in METRICS}
    bootstrap = {key: [] for key in features}
    rng = random.Random(seed)
    if n >= 2:
        for _ in range(repetitions):
            indices = [rng.randrange(n) for _ in range(n)]
            for key, values in features.items():
                bootstrap[key].append(statistics.mean(values[i] for i in indices))
    alpha = (1-confidence)/2
    def interval(values):
        return [percentile(values, alpha), percentile(values, 1-alpha)] if values else None
    methods = {}
    total_tokens = sum(r["evaluated_T"] for r in rows)
    for name in METHODS:
        methods[name] = {}
        for metric in METRICS:
            values = features[(name, metric)]
            methods[name][metric] = dict(mean=statistics.mean(values),
                confidence_interval=interval(bootstrap[(name, metric)]),
                token_weighted_mean=math.fsum(v*r["evaluated_T"] for v, r in zip(values, rows))/total_tokens)
    paired = []
    for left, right in itertools.combinations(METHODS, 2):
        differences = {}
        for metric in METRICS:
            a, b = features[(left, metric)], features[(right, metric)]
            differences[metric] = dict(mean=statistics.mean(x-y for x, y in zip(a, b)),
                confidence_interval=interval([x-y for x, y in zip(bootstrap[(left, metric)], bootstrap[(right, metric)])]))
        paired.append(dict(left=left, right=right, direction="left minus right; negative favors left", metrics=differences))
    return dict(question_count=n, evaluated_token_count=total_tokens,
                aggregation="equal question weights after averaging rounds and tokens within question",
                bootstrap=dict(repetitions=repetitions, seed=seed, confidence=confidence, unit="question",
                               interval="paired percentile; no multiplicity correction",
                               note="CI unavailable for one question; two-question smoke runs are not substantive inference"),
                methods=methods, paired_differences=paired)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references-path", type=Path, required=True)
    parser.add_argument("--estimates-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--comparison", choices=("strict", "reference-target"), default="strict")
    parser.add_argument("--token-scope", choices=("full", "project"), default="full")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Choose a fresh output directory")
    ref_meta_path = args.references_path.with_suffix(".meta.json")
    est_meta_path = args.estimates_path.with_suffix(".meta.json")
    ref_meta = json.loads(ref_meta_path.read_text(encoding="utf-8"))
    est_meta = json.loads(est_meta_path.read_text(encoding="utf-8"))
    compatibility = assess_metadata(ref_meta, est_meta, args.comparison)
    refs, estimates = read_jsonl(args.references_path), read_jsonl(args.estimates_path)
    rows, seed_audit = calculate_rows(refs, estimates, args.token_scope)
    if rows[0]["round_count"] != ref_meta["rounds"]:
        raise ValueError("Round count differs from run metadata")
    for meta in (ref_meta, est_meta):
        if meta.get("completed_questions") != len(rows):
            raise ValueError("Completed question count differs from input files")
    summary = aggregate(rows, args.bootstrap, args.seed, args.confidence)
    summary.update(schema="diagnostic_metrics_v1", status="complete", compatibility=compatibility,
        seed_audit=seed_audit, token_scope=args.token_scope,
        omitted_reference_tokens=sum(len(r["excluded_reference_token_indices"]) for r in rows),
        target="Original supplied reference target; reference advantages are never recomputed, merged or shifted",
        formulas=dict(R="mean((a-z1)*(a-z2))", A="mean((z1-block_mean(z1))*(z2-block_mean(z2)))", E="R-A"),
        negative_values="Retained without clipping or absolute value",
        input_hashes={str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in (args.references_path, args.estimates_path, ref_meta_path, est_meta_path)})
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "per_question.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    write_json(args.output_dir / "summary.json", summary)
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "metric", "question_mean", "ci_low", "ci_high", "token_weighted_mean"])
        for method in METHODS:
            for metric in METRICS:
                entry = summary["methods"][method][metric]
                ci = entry["confidence_interval"] or [None, None]
                writer.writerow([method, metric, entry["mean"], *ci, entry["token_weighted_mean"]])
    for method in METHODS:
        numbers = summary["methods"][method]
        print(f"{method}: R={numbers['R']['mean']:.8g} A={numbers['A']['mean']:.8g} E={numbers['E']['mean']:.8g}")
    print(f"comparison={args.comparison}; tokens={args.token_scope}; "
          f"omitted_reference_tokens={summary['omitted_reference_tokens']}")
    print(f"Saved metrics: {args.output_dir}")


if __name__ == "__main__":
    main()
