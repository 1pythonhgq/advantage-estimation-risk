"""Read-only checkpoint audit: parameter changes and actual train/validation predictions."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from train_diagnostic_critic import (  # Also adds the repository src path.
    PRECISION_PROTOCOL, forward_values, make_loader,
)
from diagnostic_critic_utils import read_jsonl, sha256, validate_examples


def summarize_predictions(prediction, target, train_mean):
    """Use exactly the supervised prefix positions and token weighting of training."""
    import torch
    prediction, target = prediction.double(), target.double()
    if prediction.numel() == 0 or not torch.isfinite(prediction).all():
        raise ValueError("Empty or non-finite Critic predictions")
    error = prediction - target
    target_variance = target.var(unbiased=False).item()
    quantiles = torch.quantile(prediction, torch.tensor(
        [0., .01, .5, .99, 1.], dtype=torch.float64)).tolist()
    return dict(
        prefix_count=prediction.numel(), mse=error.square().mean().item(),
        prediction_mean=prediction.mean().item(),
        prediction_std=prediction.std(unbiased=False).item(),
        prediction_quantiles=dict(zip(("min", "p01", "median", "p99", "max"), quantiles)),
        target_mean=target.mean().item(), target_variance=target_variance,
        mean_error=error.mean().item(), residual_variance=error.var(unbiased=False).item(),
        explained_variance=(None if target_variance <= 1e-12 else
                            1 - error.var(unbiased=False).item() / target_variance),
        train_mean_baseline_mse=(target - train_mean).square().mean().item(),
        zero_baseline_mse=target.square().mean().item(),
        outside_project_reward_range_fraction=((prediction < -2) | (prediction > 1)).double().mean().item(),
    )


def evaluate_predictions(critic, loader, train_mean, split):
    import torch
    predictions, targets = [], []
    critic.eval()
    with torch.inference_mode():
        for index, (ids, mask, positions, rewards, _keys) in enumerate(loader, 1):
            output = forward_values(critic, ids, mask, "cuda")
            for row, pos in enumerate(positions):
                values = output[row, pos].float().cpu()
                predictions.append(values)
                targets.append(torch.full_like(values, rewards[row].item()))
            if index == 1 or index % 100 == 0 or index == len(loader):
                print(f"{split}: {index}/{len(loader)} batches", flush=True)
    return summarize_predictions(torch.cat(predictions), torch.cat(targets), train_mean)


def parameter_changes(backbone, checkpoint):
    """Compare every backbone parameter with the same BF16-loaded SFT weights.

    The LM head is separate because value loss normally does not train it.
    This measures stored changes, not gradients or a proof of their cause.
    """
    import torch
    groups = {}
    for name, parameter in backbone.named_parameters():
        group_name = "lm_head" if name.startswith("lm_head.") else "backbone"
        group = groups.setdefault(group_name, dict(numel=0, changed=0,
            squared_change=0., squared_base=0., max_abs_change=0., dtypes=Counter()))
        stored = checkpoint["pretrained_model." + name]
        if stored.shape != parameter.shape:
            raise ValueError(f"Checkpoint shape mismatch: {name}")
        base = parameter.detach().cpu().reshape(-1)
        trained = stored.reshape(-1)
        group["dtypes"][str(stored.dtype)] += stored.numel()
        for start in range(0, base.numel(), 1_000_000):
            a = base[start:start + 1_000_000].double()
            b = trained[start:start + 1_000_000].double()
            delta = b - a
            if not torch.isfinite(delta).all():
                raise ValueError(f"Non-finite checkpoint/base parameter: {name}")
            group["numel"] += a.numel()
            group["changed"] += int(torch.count_nonzero(delta).item())
            group["squared_change"] += delta.square().sum().item()
            group["squared_base"] += a.square().sum().item()
            group["max_abs_change"] = max(group["max_abs_change"], delta.abs().max().item())
    for group in groups.values():
        group["changed_fraction"] = group["changed"] / group["numel"]
        group["rms_change"] = math.sqrt(group.pop("squared_change") / group["numel"])
        group["rms_base"] = math.sqrt(group.pop("squared_base") / group["numel"])
        group["dtypes"] = dict(group["dtypes"])
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="The best or latest_round_* subdirectory")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.output_path.exists():
        raise ValueError("Use a positive batch size and a new output path")
    training = json.loads((args.checkpoint_dir.parent / "manifest.json").read_text(encoding="utf-8"))
    data_meta = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    if data_meta.get("status") != "complete":
        raise ValueError("Prepared data is incomplete")
    if training["model_name_or_path"] != args.model_name_or_path:
        raise ValueError("Use the same original SFT model path as training")
    rows = {}
    for split in ("train", "validation"):
        path = args.data_dir / (split + ".jsonl")
        if sha256(path) != training["data_hashes"][split]:
            raise ValueError(f"{split} data differs from the training manifest")
        rows[split] = read_jsonl(path)
        validate_examples(rows[split], data_meta[split + "_source_indices"])
    train_mean = math.fsum(r["reward"] * r["response_length"] for r in rows["train"]) / sum(
        r["response_length"] for r in rows["train"])

    import torch
    from transformers import AutoTokenizer
    from treetune.models.pretrained import DIPreTrainedModelForCasualLM
    from treetune.models.pretrained_with_value_head import PreTrainedModelForValueNetwork
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference matching the existing trainer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Loading original SFT backbone and saved Critic...", flush=True)
    backbone = DIPreTrainedModelForCasualLM.from_di(
        args.model_name_or_path,
        pretrained_args={"use_flash_attention_2": True, "torch_dtype": torch.bfloat16},
        disable_dropout=True, device=torch.device("cuda"), init_base_model_only=False)
    critic = PreTrainedModelForValueNetwork(backbone).cuda()
    checkpoint = torch.load(args.checkpoint_dir / "pytorch_model.bin", map_location="cpu")
    print("Comparing all stored backbone weights against SFT...", flush=True)
    changes = parameter_changes(backbone, checkpoint)
    checkpoint_dtypes = dict(Counter(str(t.dtype) for t in checkpoint.values()))
    protocol = training.get("precision_protocol")
    if protocol == PRECISION_PROTOCOL:
        # load_state_dict otherwise silently rounds FP32 checkpoints to BF16.
        critic.float()
        critic._diagnostic_amp_bf16 = True
    elif protocol is not None:
        raise ValueError(f"Unsupported Critic precision protocol: {protocol}")
    critic.load_state_dict(checkpoint, strict=True)
    del checkpoint
    scheduler_path = args.checkpoint_dir / "scheduler.pt"
    scheduler = torch.load(scheduler_path, map_location="cpu") if scheduler_path.exists() else {}
    report = dict(schema="diagnostic_critic_audit_v1", checkpoint_dir=str(args.checkpoint_dir),
        train_target_mean_token_weighted=train_mean, parameter_changes=changes,
        checkpoint_tensor_dtypes=checkpoint_dtypes,
        precision_protocol=protocol or "legacy_direct_bf16_adamw",
        scheduler_last_lr=scheduler.get("_last_lr"), scheduler_last_epoch=scheduler.get("last_epoch"),
        training_freeze_backbone=training.get("freeze_backbone"), splits={})
    print(json.dumps({k: v for k, v in report.items() if k != "splits"}, indent=2), flush=True)
    for split in ("train", "validation"):
        loader = make_loader(rows[split], tokenizer, args.batch_size, False, 42)
        report["splits"][split] = evaluate_predictions(critic, loader, train_mean, split)
        print(json.dumps({split: report["splits"][split]}, indent=2), flush=True)
    previous = training.get("best_validation", {}).get("mse")
    if args.checkpoint_dir.name == "best" and previous is not None:
        report["validation_mse_difference_from_manifest"] = report["splits"]["validation"]["mse"] - previous
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Saved audit to {args.output_path}")


if __name__ == "__main__":
    main()
