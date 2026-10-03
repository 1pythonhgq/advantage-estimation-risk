"""Standard-library-only tests for metric formulas, joins and paired bootstrap."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from compute_diagnostic_metrics import (
    METHODS, METRICS, aggregate, assess_metadata, blocks_for, calculate_rows, read_jsonl, round_metrics,
    main,
)


def pair(row=0, rounds=1, drop_eos=False):
    reference = dict(row_index=row, source_index=100+row, selected_index=0,
                     prompt_token_ids=[9], response_token_ids=[10, 11, 0], response_text="x\u2028y",
                     finish_reason="stop", terminal_reward=1, T=3, mc_samples_per_group=2, rounds=[])
    length = 2 if drop_eos else 3
    estimate = dict(**{k: reference[k] for k in ("row_index", "source_index", "selected_index",
                    "prompt_token_ids", "response_text", "finish_reason", "terminal_reward")},
                    schema="project_methods_v2", response_token_ids=reference["response_token_ids"][:length],
                    original_response_token_ids=reference["response_token_ids"], T=length,
                    reference_token_indices=list(range(length)),
                    excluded_reference_token_indices=[2] if drop_eos else [], rounds=[])
    for rnd in range(rounds):
        groups = []
        for group in (1, 2):
            rewards = [[0, 1], [group-1, group-1], [1, 1]]
            values = [.5, float(group-1), 1, 0]
            groups.append(dict(group=group, reward_samples=rewards, values=values,
                token_advantages=[values[1]-.5, 1-values[1], 0],
                seeds=[group*100000+row*100+rnd*10+i for i in range(3)]))
        reference["rounds"].append(dict(round_index=rnd, groups=groups))
        estimate["rounds"].append(dict(round_index=rnd,
            grpo=dict(token_advantages=[.25]*length, token_advantages_unstandardized=[.1]*length,
                      seed=None if rnd == 0 else 500000+row*100+rnd*10),
            vineppo=dict(token_advantages=[.5]*length, token_step_indices=[0]*length,
                         character_boundaries=[0, 3], mc_requests=[dict(seed=300000+row*100+rnd*10)]),
            spo_int5=dict(token_advantages=[0]*length, cutpoints=[], mc_requests=[])))
    return reference, estimate


class MetricsTests(unittest.TestCase):
    def test_exact_small_example_and_decomposition(self):
        result = round_metrics([2, 2], [1, 3], [2, 4], [[0, 1]])
        self.assertEqual(result, dict(R=1, A=1, E=0))
        pointwise = round_metrics([2, 2], [1, 3], [2, 4], [[0], [1]])
        self.assertEqual(pointwise, dict(R=1, A=0, E=1))

    def test_negative_unbiased_estimates_are_preserved(self):
        result = round_metrics([0, 0], [1, -1], [-1, 1], [[0, 1]])
        self.assertEqual(result, dict(R=-1, A=-1, E=0))

    def test_block_means_not_advantage_or_token_means(self):
        with self.assertRaisesRegex(ValueError, "not constant"):
            round_metrics([1, 2], [0, 1], [0, 1], [[0, 1]])
        with self.assertRaisesRegex(ValueError, "cover every"):
            round_metrics([1, 1], [0, 1], [0, 1], [[0], [0]])

    def test_vine_uses_actual_token_step_ids_and_spo_off_by_one(self):
        vine = dict(character_boundaries=[0, 1, 3, 5], token_step_indices=[0, 2, 2])
        self.assertEqual(blocks_for("vineppo", vine, 3), [[0], [1, 2]])
        self.assertEqual(blocks_for("spo_int5", dict(cutpoints=[-1, 1, 3]), 6), [[0, 1], [2, 3], [4, 5]])
        self.assertEqual(blocks_for("spo_int5", dict(cutpoints=[]), 3), [[0, 1, 2]])
        self.assertEqual(blocks_for("grpo", {}, 3), [[0, 1, 2]])

    def test_unordered_rows_and_rounds_join_by_identity(self):
        r0, a0 = pair(0, 2)
        r1, a1 = pair(1, 2)
        a0["rounds"].reverse()
        r1["rounds"].reverse()
        result, audit = calculate_rows([r1, r0], [a0, a1], "full")
        self.assertEqual([r["row_index"] for r in result], [0, 1])
        self.assertTrue(audit["no_recorded_seed_overlap"])
        for r in result:
            self.assertEqual(r["methods"]["grpo"]["A"], r["methods"]["grpo_unstandardized"]["A"])

    def test_round_cross_products_are_averaged_after_computation(self):
        ref, est = pair(rounds=2)
        est["rounds"][0]["grpo"]["token_advantages"] = [1]*3
        est["rounds"][1]["grpo"]["token_advantages"] = [-1]*3
        rows, _ = calculate_rows([ref], [est], "full")
        risks = [r["methods"]["grpo"]["R"] for r in rows[0]["rounds"]]
        self.assertEqual(rows[0]["methods"]["grpo"]["R"], sum(risks)/2)
        z1, z2 = [g["token_advantages"] for g in ref["rounds"][0]["groups"]]
        wrong = round_metrics([0]*3, z1, z2, [[0, 1, 2]])["R"]
        self.assertAlmostEqual(rows[0]["methods"]["grpo"]["R"]-wrong, 1)

    def test_explicit_eos_scope_does_not_rewrite_reference(self):
        ref, est = pair(drop_eos=True)
        before = copy.deepcopy(ref)
        with self.assertRaisesRegex(ValueError, "terminal reference token"):
            calculate_rows([ref], [est], "full")
        rows, _ = calculate_rows([ref], [est], "project")
        self.assertEqual(rows[0]["evaluated_T"], 2)
        self.assertEqual(rows[0]["excluded_reference_token_indices"], [2])
        self.assertEqual(ref, before)

    def test_mismatch_and_shared_reference_seeds_fail(self):
        ref, est = pair()
        bad = copy.deepcopy(est)
        bad["selected_index"] = 1
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            calculate_rows([ref], [bad], "full")
        with self.assertRaisesRegex(ValueError, "Question sets differ"):
            calculate_rows([ref], [pair(1)[1]], "full")
        ref["rounds"][0]["groups"][1]["seeds"] = ref["rounds"][0]["groups"][0]["seeds"]
        with self.assertRaisesRegex(ValueError, "reuse seeds"):
            calculate_rows([ref], [est], "full")

    def test_estimator_reference_seed_overlap_fails(self):
        ref, est = pair()
        est["rounds"][0]["vineppo"]["mc_requests"][0]["seed"] = ref["rounds"][0]["groups"][0]["seeds"][0]
        with self.assertRaisesRegex(ValueError, "overlap reference"):
            calculate_rows([ref], [est], "full")

    def test_reference_corruption_is_detected(self):
        ref, est = pair()
        ref["rounds"][0]["groups"][0]["token_advantages"][0] = 999
        with self.assertRaisesRegex(ValueError, "identity failed"):
            calculate_rows([ref], [est], "full")

    def test_metadata_strict_versus_explicit_reference_target(self):
        shared = dict(status="complete", trajectory_metadata={}, reward_protocol={}, gamma=1, rounds=1)
        reference = dict(**shared, input_sha256="abc")
        estimate = dict(**shared, trajectory_sha256="abc", schema="project_methods_v2",
                        probability_mask=False, batch_whitening=False)
        with self.assertRaisesRegex(ValueError, "Identical target"):
            assess_metadata(reference, estimate, "strict")
        self.assertFalse(assess_metadata(reference, estimate, "reference-target")["identical_target_protocol_verified"])

    def test_bootstrap_is_paired_and_question_weighted(self):
        rows = []
        for length, value in ((1, 0), (9, 10), (2, 5)):
            rows.append(dict(evaluated_T=length, methods={name: {metric: value+(2 if name == "vineppo" else 0)
                                              for metric in METRICS} for name in METHODS}))
        summary = aggregate(rows, 100, 42)
        self.assertEqual(summary, aggregate(rows, 100, 42))
        self.assertEqual(summary["methods"]["grpo"]["R"]["mean"], 5)
        self.assertAlmostEqual(summary["methods"]["grpo"]["R"]["token_weighted_mean"], 100/12)
        difference = next(d for d in summary["paired_differences"] if d["left"] == "grpo" and d["right"] == "vineppo")
        for endpoint in difference["metrics"]["R"]["confidence_interval"]:
            self.assertAlmostEqual(endpoint, -2)
        self.assertIsNone(aggregate(rows[:1], 100)["methods"]["grpo"]["R"]["confidence_interval"])

    def test_jsonl_unicode_separator_is_not_a_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps({"text": "a\u2028b"}, ensure_ascii=False)+"\n", encoding="utf-8")
            self.assertEqual(read_jsonl(path), [{"text": "a\u2028b"}])

    def test_cli_writes_self_describing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref_path, est_path = root / "ref.jsonl", root / "est.jsonl"
            refs, estimates = zip(pair(0), pair(1))
            for path, records in ((ref_path, refs), (est_path, estimates)):
                path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in records), encoding="utf-8")
            shared = dict(status="complete", trajectory_metadata={}, reward_protocol={}, gamma=1,
                          rounds=1, completed_questions=2)
            ref_meta = dict(**shared, input_sha256="abc")
            est_meta = dict(**shared, trajectory_sha256="abc", schema="project_methods_v2",
                            probability_mask=False, batch_whitening=False)
            for path, meta in ((ref_path, ref_meta), (est_path, est_meta)):
                path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
            out = root / "metrics"
            with patch("sys.argv", ["metrics", "--references-path", str(ref_path), "--estimates-path", str(est_path),
                      "--output-dir", str(out), "--comparison", "reference-target", "--bootstrap", "10"]):
                main()
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["question_count"], 2)
            self.assertEqual(summary["compatibility"]["comparison"], "reference-target")
            self.assertEqual(len(read_jsonl(out / "per_question.jsonl")), 2)
            self.assertTrue((out / "summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
