"""Compute project actor probabilities and VinePPO/SPO-int5 partitions.

Teacher-forced actor forward only; generation remains entirely in vLLM.
Run this process to completion BEFORE starting the estimator rollout process.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from compute_token_reference_advantages_vllm import load_jsonl_records, validate_record, write_json
from diagnostic_reference_utils import project_reward_function
from project_method_adapter import (SCHEMA, ProjectMethods, load_project_protocol,
                                    source_fingerprint, validate_sampling_metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    meta_path = args.trajectories_path.with_suffix(".meta.json")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    required = ("model_name_or_path", "inference_engine", "vllm_version", "temperature",
                "top_p", "max_new_tokens", "stop_text", "seed", "dtype",
                "enable_prefix_caching", "gpu_memory_utilization", "max_model_len")
    if any(key not in metadata for key in required) or metadata["inference_engine"] != "vllm.LLM":
        raise ValueError("Missing or incompatible trajectory engine metadata")
    if not math.isfinite(metadata["temperature"]) or metadata["temperature"] <= 0:
        raise ValueError("Actor scoring requires a positive finite temperature")
    out_meta = args.output_path.with_suffix(".meta.json")
    partial = args.output_path.with_suffix(".partial.jsonl")
    if args.output_path.suffix != ".jsonl":
        raise ValueError("Use .jsonl output")
    for path in (args.output_path, out_meta, partial):
        if path.exists() or path.resolve() in (args.trajectories_path.resolve(), meta_path.resolve()):
            raise FileExistsError(f"Choose a fresh output path: {path}")
    records = load_jsonl_records(args.trajectories_path, args.limit)
    if not records or len({r['row_index'] for r in records}) != len(records):
        raise ValueError("Empty or duplicate trajectory rows")
    protocol = load_project_protocol(metadata["seed"])
    validate_sampling_metadata(metadata, protocol)
    project_reward_function()  # Also adds repository src to sys.path.
    import torch
    from transformers import AutoTokenizer
    from treetune.models.pretrained import DIPreTrainedModelForCasualLM
    from treetune.trainers.ppo_trainer import PPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(metadata["model_name_or_path"], use_fast=True,
                                             trust_remote_code=metadata.get("trust_remote_code", False))
    methods = ProjectMethods(tokenizer, protocol)
    bases = {}
    for record in records:
        validate_record(record, tokenizer, metadata)
        try:
            bases[record["row_index"]] = methods.base_trajectory(record)
            methods.vine_partition(bases[record["row_index"]])
        except (ValueError, AssertionError) as exc:
            raise ValueError(f"Vine partition failed for row_index={record['row_index']}: {exc}") from exc
    # Rho GSM8K actor configuration: BF16, FlashAttention2, dropout disabled.
    model = DIPreTrainedModelForCasualLM.from_di(
        hf_model_name=metadata["model_name_or_path"], disable_dropout=True,
        pretrained_args={"use_flash_attention_2": True, "torch_dtype": torch.bfloat16},
        device=torch.device("cuda"),
    ).eval()
    actor_context = SimpleNamespace(ppo_hparams=SimpleNamespace(temperature=metadata["temperature"]))
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "running", "input_sha256": hashlib.sha256(args.trajectories_path.read_bytes()).hexdigest(),
                "trajectory_metadata": metadata, "actor_dtype": "bfloat16", "actor_flash_attention_2": True,
                "probability_source": "PPOTrainer._forward_pass_actor: float(logits)/temperature -> log_softmax; no top-p",
                "schema": SCHEMA, "project_protocol": protocol, "source_fingerprint": source_fingerprint(),
                "variant": "project_estimation_without_spo_probability_mask"}
    write_json(out_meta, manifest)
    try:
        with partial.open("w", encoding="utf-8") as handle, torch.inference_mode():
            for record in records:
                base = bases[record["row_index"]]
                prompt_ids = base["query_token_ids"]
                response_ids = base["response_token_ids"]
                ids = torch.tensor([prompt_ids + response_ids], device="cuda", dtype=torch.long)
                inputs = {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}
                outputs = PPOTrainer._forward_pass_actor(actor_context, model, inputs, return_all_logps=True)
                logprobs = outputs["all_logps"][0, len(prompt_ids)-1:].cpu().tolist()
                if len(logprobs) != len(response_ids):
                    raise ValueError("Actor response logprob alignment failed")
                result = methods.prepare(record, logprobs)
                handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                print(f"row={record['row_index']} tokens={len(response_ids)} "
                      f"vine_steps={len(result['vineppo']['steps'])} "
                      f"spo_cutpoints={len(result['spo_int5']['cutpoints'])} "
                      f"excluded_reference_tokens={result['excluded_reference_token_indices']}", flush=True)
                del outputs, inputs, ids
        partial.replace(args.output_path)
        manifest["status"] = "complete"
        write_json(out_meta, manifest)
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        write_json(out_meta, manifest)
        raise


if __name__ == "__main__":
    main()
