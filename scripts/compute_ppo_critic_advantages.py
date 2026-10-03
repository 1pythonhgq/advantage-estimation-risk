"""Compute PPO GAE advantages with the trained diagnostic Critic."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compute_token_reference_advantages_vllm import load_jsonl_records, write_json
from diagnostic_reference_utils import REWARD_PROTOCOL, score_response
from train_diagnostic_critic import PRECISION_PROTOCOL, forward_values


def finite_range(name, value, low=0.0, high=1.0):
    if not math.isfinite(value) or value < low or value > high:
        raise ValueError(f'Invalid {name}: {value}')


def compute_gae(values, terminal_reward, gamma=1.0, lam=1.0):
    """Match PPOTrainer._compute_advantages for one unpadded response."""
    if not values or not all(math.isfinite(float(x)) for x in values):
        raise ValueError('Critic values must be finite and non-empty')
    finite_range('gamma', gamma)
    finite_range('lam', lam)
    if not math.isfinite(float(terminal_reward)):
        raise ValueError('Terminal reward must be finite')
    rewards = [0.0] * len(values)
    rewards[-1] = float(terminal_reward)
    last_gae = 0.0
    reversed_advantages = []
    for t in reversed(range(len(values))):
        next_value = float(values[t + 1]) if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value - float(values[t])
        last_gae = delta + gamma * lam * last_gae
        reversed_advantages.append(last_gae)
    advantages = list(reversed(reversed_advantages))
    returns = [a + float(v) for a, v in zip(advantages, values)]
    no_bias_reversed = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        no_bias_reversed.append(running)
    return advantages, returns, list(reversed(no_bias_reversed)), rewards


def whiten(values, epsilon=1e-8):
    if not values:
        return []
    mean = math.fsum(values) / len(values)
    variance = math.fsum((x - mean) ** 2 for x in values) / max(1, len(values) - 1)
    scale = math.sqrt(variance + epsilon)
    return [(x - mean) / scale for x in values]


def make_examples(records, all_rollouts=False):
    examples = []
    for i, record in enumerate(records):
        prompt_ids = list(record.get('prompt_token_ids', []))
        rollouts = record.get('rollouts')
        selected = record.get('selected_index')
        if not prompt_ids or not isinstance(rollouts, list) or not rollouts:
            raise ValueError(f'Invalid prompt/rollouts at row {i}')
        if selected is None or not 0 <= selected < len(rollouts):
            raise ValueError(f'Invalid selected_index at row {i}')
        indices = range(len(rollouts)) if all_rollouts else [selected]
        for rollout_index in indices:
            rollout = rollouts[rollout_index]
            response_ids = list(rollout.get('response_token_ids', []))
            if not response_ids:
                raise ValueError(f'Empty response at row {i}, rollout {rollout_index}')
            reward = rollout.get('reward')
            if reward is None:
                reward = score_response(rollout['response_text'], record['gold_answer'], rollout['finish_reason'])
            examples.append({
                'row_index': record.get('row_index', i),
                'source_index': record.get('source_index'),
                'rollout_index': rollout_index,
                'selected': rollout_index == selected,
                'input_ids': prompt_ids + response_ids,
                'prompt_length': len(prompt_ids),
                'response_length': len(response_ids),
                'response_token_ids': response_ids,
                'terminal_reward': float(reward),
                'response_text': rollout.get('response_text'),
                'finish_reason': rollout.get('finish_reason'),
            })
    return examples


def load_critic(args, device):
    import torch
    from treetune.models.pretrained import DIPreTrainedModelForCasualLM
    from treetune.models.pretrained_with_value_head import PreTrainedModelForValueNetwork
    manifest_path = args.critic_checkpoint_dir.parent / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f'Missing critic manifest: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('model_name_or_path') != args.model_name_or_path:
        raise ValueError('Critic and SFT model paths differ')
    if manifest.get('precision_protocol') != PRECISION_PROTOCOL:
        raise ValueError(f'Unsupported Critic precision protocol: {manifest.get("precision_protocol")!r}')
    backbone = DIPreTrainedModelForCasualLM.from_di(
        args.model_name_or_path,
        pretrained_args={'use_flash_attention_2': True, 'torch_dtype': torch.bfloat16},
        disable_dropout=True, device=device, init_base_model_only=False)
    critic = PreTrainedModelForValueNetwork(backbone).to(device=device, dtype=torch.float32)
    critic._diagnostic_amp_bf16 = True
    checkpoint = args.critic_checkpoint_dir / 'pytorch_model.bin'
    if not checkpoint.exists():
        raise FileNotFoundError(f'Missing critic checkpoint: {checkpoint}')
    critic.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)
    critic.eval()
    return critic, manifest


def predict_values(critic, examples, batch_size, device, pad_token_id):
    import torch
    values = {}
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            length = max(len(x['input_ids']) for x in batch)
            ids = torch.tensor([x['input_ids'] + [pad_token_id] * (length - len(x['input_ids'])) for x in batch], dtype=torch.long)
            mask = torch.tensor([[1] * len(x['input_ids']) + [0] * (length - len(x['input_ids'])) for x in batch], dtype=torch.long)
            output = forward_values(critic, ids, mask, device)
            for row, example in enumerate(batch):
                start_pos = example['prompt_length'] - 1
                end_pos = example['prompt_length'] + example['response_length'] - 1
                vector = output[row, start_pos:end_pos].float().cpu().tolist()
                if len(vector) != example['response_length']:
                    raise ValueError('Critic value span does not match response length')
                values[(example['row_index'], example['rollout_index'])] = vector
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectories-path', type=Path, required=True)
    parser.add_argument('--critic-checkpoint-dir', type=Path, required=True)
    parser.add_argument('--model-name-or-path', required=True)
    parser.add_argument('--output-path', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--lam', type=float, default=1.0)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--whiten-advantages', action='store_true')
    parser.add_argument('--all-rollouts', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    finite_range('gamma', args.gamma)
    finite_range('lam', args.lam)
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError('Invalid batch size or limit')
    if args.output_path.suffix != '.jsonl':
        raise ValueError('Use a .jsonl output path')
    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(f'Output exists: {args.output_path}')
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('PPO Critic inference requires CUDA BF16 support')
    metadata_path = args.trajectories_path.with_suffix('.meta.json')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
    records = load_jsonl_records(args.trajectories_path, args.limit)
    examples = make_examples(records, args.all_rollouts)
    device = torch.device('cuda')
    critic, critic_manifest = load_critic(args, device)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    values = predict_values(critic, examples, args.batch_size, device, tokenizer.pad_token_id)
    rows = []
    all_advantages = []
    for example in examples:
        key = (example['row_index'], example['rollout_index'])
        raw, returns, no_bias, rewards = compute_gae(values[key], example['terminal_reward'], args.gamma, args.lam)
        all_advantages.extend(raw)
        rows.append({**example, 'critic_values': values[key], 'rewards': rewards,
                     'raw_token_advantages': raw, 'returns': returns, 'no_bias_returns': no_bias})
    normalized = whiten(all_advantages) if args.whiten_advantages else None
    offset = 0
    for row in rows:
        n = len(row['raw_token_advantages'])
        row['token_advantages'] = normalized[offset:offset + n] if normalized is not None else row['raw_token_advantages']
        row['whitening'] = 'global_masked_unbiased_epsilon_1e-8' if normalized is not None else 'none'
        offset += n
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output_meta = args.output_path.with_suffix('.meta.json')
    manifest = {
        'schema': 'ppo_critic_advantages_v1', 'status': 'complete',
        'input_path': str(args.trajectories_path.resolve()),
        'input_sha256': hashlib.sha256(args.trajectories_path.read_bytes()).hexdigest(),
        'trajectory_metadata': metadata, 'critic_checkpoint': str(args.critic_checkpoint_dir.resolve()),
        'critic_manifest': critic_manifest, 'gamma': args.gamma, 'lam': args.lam,
        'whiten_advantages': args.whiten_advantages, 'all_rollouts': args.all_rollouts,
        'reward_protocol': REWARD_PROTOCOL,
        'formula': 'PPOTrainer._compute_advantages; terminal reward; V(s_T)=0',
        'record_count': len(rows), 'token_count': sum(len(x['raw_token_advantages']) for x in rows),
    }
    with args.output_path.open('w' if args.overwrite else 'x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    write_json(output_meta, manifest)
    print(f'Saved PPO Critic advantages: {args.output_path}', flush=True)


if __name__ == '__main__':
    main()
