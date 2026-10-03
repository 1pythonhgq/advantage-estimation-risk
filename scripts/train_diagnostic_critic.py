"""Train the diagnostic Critic with the project's PPO value-loss protocol.

The Actor is fixed.  Each round refreshes the old values, trains for the
project's two PPO value epochs, evaluates on a fixed question-disjoint
validation split, and keeps the best checkpoint plus the two latest
resumable checkpoints.

This is a standalone offline trainer, not PPOTrainer/DeepSpeed. Trainable
parameters and AdamW state are FP32; CUDA BF16 autocast is used for forwards.
Updating BF16 parameter storage directly loses small learning-rate updates.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

# The repository uses a ``src`` layout and is not necessarily installed as a
# package in the training environment.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from diagnostic_critic_utils import RegressionStatistics, read_jsonl, sha256, validate_examples, write_json

PRECISION_PROTOCOL = "fp32_parameters_adamw_bf16_autocast_v1"


def forward_values(model, ids, mask, device):
    """Use the same forward precision for training, cached values, and audits.

    Absence of the flag preserves the legacy BF16-backbone/FP32-head audit.
    The new protocol autocasts linear operations, including the value head,
    to BF16; value-loss arithmetic is explicitly promoted to FP32.
    """
    import torch
    with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16,
                        enabled=getattr(model, "_diagnostic_amp_bf16", False)):
        return model(input_ids=ids.to(device), attention_mask=mask.to(device),
                     return_dict=True, use_cache=False)


def constant_baseline(examples, train_mean):
    count = sum(row["response_length"] for row in examples)
    return math.fsum(row["response_length"] * (row["reward"] - train_mean) ** 2
                     for row in examples) / count


def validate_resume_manifest(previous, current):
    # A legacy BF16 optimizer must not silently restore over the corrected
    # optimizer. A different horizon also invalidates the saved LR schedule.
    keys = ("precision_protocol", "model_name_or_path", "data_hashes", "seed",
            "max_rounds", "epochs_per_round", "learning_rate", "batch_size",
            "gradient_accumulation_steps", "warmup_ratio", "max_grad_norm",
            "cliprange_value", "freeze_backbone", "gradient_checkpointing",
            "patience", "min_delta")
    for key in keys:
        if previous.get(key) != current.get(key):
            raise ValueError(f"Cannot resume with different {key}; use a new output directory")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-name-or-path",
                        default="/root/autodl-tmp/models/rho-1b-sft-GSM8K")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/root/autodl-tmp/critic_models/rho1b_gsm8k"))
    parser.add_argument("--max-rounds", type=int, default=100)
    parser.add_argument("--epochs-per-round", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--cliprange-value", type=float, default=.2)
    parser.add_argument("--patience", type=int, default=5,
                        help="Stop after this many consecutive rounds without validation-MSE improvement")
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Train only the scalar value head; default trains the PPO-shaped critic end to end")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the newest checkpoint under output-dir")
    return parser.parse_args()


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate(batch, pad_token_id):
    import torch
    length = max(len(x["input_ids"]) for x in batch)
    ids, mask, positions, targets, keys = [], [], [], [], []
    for example in batch:
        seq = example["input_ids"]
        padding = length-len(seq)
        ids.append(seq+[pad_token_id]*padding)
        mask.append([1]*len(seq)+[0]*padding)
        start, end = example["prompt_length"]-1, example["prompt_length"]+example["response_length"]-1
        positions.append(list(range(start, end)))
        targets.append(float(example["reward"]))
        keys.append(f'{example["source_index"]}:{example["rollout_index"]}')
    return (torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long),
            positions, torch.tensor(targets, dtype=torch.float32), keys)


def _select_values(outputs, positions, row):
    if not positions:
        raise ValueError("Empty value supervision span")
    return outputs[row, positions].float()


def forward_value_loss(model, ids, mask, positions, targets, old_values,
                       device, cliprange_value):
    import torch
    outputs = forward_values(model, ids, mask, device)
    vf_losses = []
    for row, pos in enumerate(positions):
        values = _select_values(outputs, pos, row)
        target = targets[row].to(device).expand_as(values)
        old = old_values[row].to(device)
        clipped = torch.clamp(values, old - cliprange_value, old + cliprange_value)
        loss_1 = (values - target) ** 2
        loss_2 = (clipped - target) ** 2
        vf_losses.append(torch.maximum(loss_1, loss_2))
    # This is exactly PPOTrainer._compute_critics_loss's 0.5 * masked_mean;
    # every supervised response-token state contributes one term.
    return 0.5 * torch.cat(vf_losses).mean()


def predict_values(model, loader, device):
    import torch
    model.eval()
    result = {}
    with torch.inference_mode():
        for batch in loader:
            ids, mask, positions, _targets, keys = batch
            outputs = forward_values(model, ids, mask, device)
            for row, (pos, key) in enumerate(zip(positions, keys)):
                result[key] = outputs[row, pos].float().detach().cpu()
    return result


def evaluate(model, loader, device):
    import torch
    model.eval()
    stats = RegressionStatistics()
    with torch.inference_mode():
        for batch in loader:
            ids, mask, positions, targets, keys = batch
            outputs = forward_values(model, ids, mask, device)
            pred = []
            for row, pos in enumerate(positions):
                values = _select_values(outputs, pos, row)
                pred.append(values)
            for prediction, target in zip(pred, targets):
                n = prediction.numel()
                mean_e = float((prediction-torch.as_tensor(target, device=device)).mean().item())
                mean_y = float(target.item())
                errors = prediction - target.to(device)
                m2_e = float(((errors-errors.mean())**2).sum().item())
                # The target is constant over every prefix in this example.
                stats.add(n, mean_y, 0.0, mean_e, m2_e, float(((prediction-target.to(device))**2).sum().item()))
    return stats.result()


def make_loader(examples, tokenizer, batch_size, shuffle, seed):
    import torch
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(examples, batch_size=batch_size, shuffle=shuffle,
        # PPO's training loader drops an incomplete micro-batch; validation
        # evaluates every example.
        drop_last=shuffle, generator=generator,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id))


def checkpoint_dirs(output_dir):
    return sorted((p for p in output_dir.glob("latest_round_*") if p.is_dir()),
                  key=lambda p: int(p.name.rsplit("_", 1)[1]))


def save_checkpoint(path, critic, optimizer, scheduler, round_index, bad_rounds,
                    best_mse, history, tokenizer, global_step, optimizer_steps):
    import torch
    path.mkdir(parents=True, exist_ok=True)
    torch.save(critic.state_dict(), path / "pytorch_model.bin")
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    torch.save(scheduler.state_dict(), path / "scheduler.pt")
    (path / "state.json").write_text(json.dumps({
        "round": round_index, "bad_rounds": bad_rounds,
        "best_mse": best_mse, "history": history,
        "global_step": global_step, "optimizer_steps": optimizer_steps,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    tokenizer.save_pretrained(path)


def load_checkpoint(path, critic, optimizer, scheduler, tokenizer):
    import torch
    critic.load_state_dict(torch.load(path / "pytorch_model.bin", map_location="cpu"))
    optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location="cpu"))
    scheduler.load_state_dict(torch.load(path / "scheduler.pt", map_location="cpu"))
    state = json.loads((path / "state.json").read_text(encoding="utf-8"))
    return state


def main():
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be combined")
    if (not args.resume and args.output_dir.exists()
            and any(args.output_dir.iterdir()) and not args.overwrite):
        raise FileExistsError("Output directory is not empty; choose a new path or --overwrite")
    if (not 1 <= args.max_rounds <= 100 or not 1 <= args.epochs_per_round
            or args.learning_rate <= 0 or args.batch_size <= 0
            or args.gradient_accumulation_steps <= 0 or args.patience < 1):
        raise ValueError("Invalid fixed training budget; max_rounds must be in [1, 100]")
    data_manifest = json.loads((args.data_dir/"manifest.json").read_text(encoding="utf-8"))
    if data_manifest.get("schema") != "diagnostic_critic_data_v1" or data_manifest.get("status") != "complete":
        raise ValueError("Prepared data is missing or incomplete")
    train, validation = read_jsonl(args.data_dir/"train.jsonl"), read_jsonl(args.data_dir/"validation.jsonl")
    validate_examples(train, data_manifest["train_source_indices"])
    validate_examples(validation, data_manifest["validation_source_indices"])
    if set(data_manifest["train_source_indices"]) & set(data_manifest["validation_source_indices"]):
        raise ValueError("Question leakage between train and validation")
    if args.output_dir.resolve() == args.data_dir.resolve():
        raise ValueError("Training output must be separate from prepared data")
    seed_everything(args.seed)
    import torch
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from treetune.models.pretrained import DIPreTrainedModelForCasualLM
    from treetune.models.pretrained_with_value_head import PreTrainedModelForValueNetwork
    if not torch.cuda.is_available():
        raise RuntimeError("Critic training requires CUDA to match the project's PPO setup")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This training precision protocol requires CUDA BF16 support")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # This is the same constructor used by the project's
    # ``pretrained_causal_lm_with_value_head`` PPO critic: a causal-LM
    # backbone (with its LM head) wrapped by the scalar value head.
    backbone = DIPreTrainedModelForCasualLM.from_di(
        args.model_name_or_path,
        pretrained_args={"use_flash_attention_2": True, "torch_dtype": torch.bfloat16},
        disable_dropout=True,
        device=device,
        init_base_model_only=False,
    )
    # Preserve the initial BF16-loaded SFT snapshot, but accumulate optimizer
    # updates in FP32 instead of rounding each update into BF16 storage.
    critic = PreTrainedModelForValueNetwork(backbone).to(device=device, dtype=torch.float32)
    critic._diagnostic_amp_bf16 = True
    critic.train()
    if args.gradient_checkpointing:
        critic.gradient_checkpointing_enable()
    if args.freeze_backbone:
        for parameter in critic.pretrained_model.parameters():
            parameter.requires_grad_(False)
    train_loader = make_loader(train, tokenizer, args.batch_size, True, args.seed)
    if not len(train_loader):
        raise ValueError("Training data must contain at least one complete microbatch")
    train_value_loader = make_loader(train, tokenizer, args.batch_size, False, args.seed)
    validation_loader = make_loader(validation, tokenizer, args.batch_size, False, args.seed)
    parameters = [p for p in critic.parameters() if p.requires_grad]
    if any(p.dtype != torch.float32 for p in parameters):
        raise RuntimeError("AdamW requires FP32 parameter storage in this trainer")
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0,
                                 betas=(0.9, 0.999), eps=1e-8, foreach=False)
    updates_per_round = max(1, math.ceil(len(train_loader)/args.gradient_accumulation_steps))
    total_updates = args.max_rounds * args.epochs_per_round * updates_per_round
    warmup_steps = round(total_updates*args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)
    best_mse, bad_rounds = float("inf"), 0
    history, global_step, optimizer_steps, start_round = [], 0, 0, 0
    train_mean = math.fsum(r["reward"] * r["response_length"] for r in train) / sum(
        r["response_length"] for r in train)
    baselines = {"train": constant_baseline(train, train_mean),
                 "validation": constant_baseline(validation, train_mean)}
    manifest = dict(schema="diagnostic_critic_v1", status="running", model_name_or_path=args.model_name_or_path,
        backbone_constructor="DIPreTrainedModelForCasualLM.from_di(init_base_model_only=False)",
        value_model="PreTrainedModelForValueNetwork", disable_dropout=True,
        dtype="torch.float32", forward_dtype="torch.bfloat16", value_head_dropout=None,
        precision_protocol=PRECISION_PROTOCOL,
        optimizer="torch.optim.AdamW", optimizer_betas=[0.9, 0.999], optimizer_eps=1e-8,
        train_mean_baseline_mse=baselines, train_target_mean=train_mean,
        data_manifest=data_manifest, data_hashes={s: sha256(args.data_dir/(s+".jsonl")) for s in ("train", "validation")},
        seed=args.seed, max_rounds=args.max_rounds, epochs_per_round=args.epochs_per_round,
        learning_rate=args.learning_rate, batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps, warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm, cliprange_value=args.cliprange_value,
        patience=args.patience, min_delta=args.min_delta, eval_every_rounds=1,
        freeze_backbone=args.freeze_backbone, gradient_checkpointing=args.gradient_checkpointing,
        objective="project PPO clipped value loss with terminal project reward targets",
        value_alignment="critic[P-1:P+T-1] predicts s0..s(T-1); V(terminal)=0", history=history)
    # Check compatibility before replacing the previous run's manifest.
    if args.resume:
        previous = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
        validate_resume_manifest(previous, manifest)
        latest = checkpoint_dirs(args.output_dir)
        if not latest:
            raise FileNotFoundError("--resume requested but no latest_round_* checkpoint exists")
        state = load_checkpoint(latest[-1], critic, optimizer, scheduler, tokenizer)
        start_round = int(state["round"])
        bad_rounds = int(state["bad_rounds"])
        best_mse = float(state["best_mse"])
        history = list(state["history"])
        global_step = int(state.get("global_step", 0))
        optimizer_steps = int(state.get("optimizer_steps", 0))
        manifest["initial_validation"] = previous.get("initial_validation")
        if start_round >= args.max_rounds or bad_rounds >= args.patience:
            raise ValueError("Saved run already reached its round limit or early stopping condition")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        if not args.resume:
            manifest["initial_validation"] = evaluate(critic, validation_loader, device)
        write_json(args.output_dir / "manifest.json", {**manifest, "history": history})
        print(json.dumps({"precision_protocol": PRECISION_PROTOCOL,
                          "train_mean_baseline_mse": baselines,
                          "initial_validation": manifest["initial_validation"]}), flush=True)

        for round_index in range(start_round, args.max_rounds):
            # PPO uses values from the beginning of this iteration as old_values.
            old_value_map = predict_values(critic, train_value_loader, device)
            critic.train()
            optimizer.zero_grad(set_to_none=True)
            micro_steps = 0
            for epoch in range(args.epochs_per_round):
                critic.train()
                # Round-boundary resumes reproduce the same subsequent batches.
                train_loader.generator.manual_seed(
                    args.seed + round_index * args.epochs_per_round + epoch)
                for batch_index, batch in enumerate(train_loader):
                    ids, mask, positions, targets, keys = batch
                    old_values = [old_value_map[k] for k in keys]
                    loss = forward_value_loss(
                        critic, ids, mask, positions, targets, old_values,
                        device, args.cliprange_value,
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite Critic value loss")
                    group_start = (batch_index // args.gradient_accumulation_steps
                                   * args.gradient_accumulation_steps)
                    accumulation_count = min(args.gradient_accumulation_steps,
                                             len(train_loader) - group_start)
                    (loss / accumulation_count).backward()
                    global_step += 1
                    micro_steps += 1
                    if (micro_steps >= args.gradient_accumulation_steps
                            or batch_index + 1 == len(train_loader)):
                        torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm,
                                                       error_if_nonfinite=True)
                        last_update_lr = optimizer.param_groups[0]["lr"]
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        optimizer_steps += 1
                        micro_steps = 0
            train_metrics = evaluate(critic, train_value_loader, device)
            metrics = evaluate(critic, validation_loader, device)
            improved = metrics["mse"] < best_mse - args.min_delta
            if improved:
                best_mse, bad_rounds = metrics["mse"], 0
            else:
                bad_rounds += 1
            history.append(dict(round=round_index + 1, optimizer_step=optimizer_steps,
                                bad_rounds=bad_rounds, train=train_metrics,
                                last_update_lr=last_update_lr,
                                next_update_lr=optimizer.param_groups[0]["lr"], **metrics))
            print(json.dumps(history[-1]), flush=True)
            if improved:
                save_checkpoint(args.output_dir / "best", critic, optimizer, scheduler,
                                round_index + 1, bad_rounds, best_mse, history,
                                tokenizer, global_step, optimizer_steps)
            latest_path = args.output_dir / f"latest_round_{round_index + 1:04d}"
            save_checkpoint(latest_path, critic, optimizer, scheduler, round_index + 1,
                            bad_rounds, best_mse, history, tokenizer,
                            global_step, optimizer_steps)
            latest = checkpoint_dirs(args.output_dir)
            for stale in latest[:-2]:
                import shutil
                shutil.rmtree(stale)
            write_json(args.output_dir / "manifest.json", {**manifest, "status": "running",
                        "round": round_index + 1, "best_mse": best_mse,
                        "bad_rounds": bad_rounds, "history": history,
                        "optimizer_steps": optimizer_steps})
            if bad_rounds >= args.patience:
                break
        best_path = args.output_dir / "best"
        if best_path.exists():
            import torch
            critic.load_state_dict(torch.load(best_path / "pytorch_model.bin", map_location="cpu"))
        final = evaluate(critic, validation_loader, device)
        manifest.update(status="complete", elapsed_seconds=time.monotonic()-started,
                        best_validation=final, best_mse=best_mse, bad_rounds=bad_rounds,
                        history=history,
                        stopped_round=history[-1]["round"] if history else start_round,
                        optimizer_steps=optimizer_steps, parameter_count=sum(p.numel() for p in parameters))
        torch.save(critic.state_dict(), args.output_dir / "pytorch_model.bin")
        tokenizer.save_pretrained(args.output_dir)
        write_json(args.output_dir/"manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="failed", error=str(exc), elapsed_seconds=time.monotonic()-started, history=history)
        write_json(args.output_dir/"manifest.json", manifest)
        raise
    print(f"Saved diagnostic critic to {args.output_dir}; validation={final}")


if __name__ == "__main__":
    main()
