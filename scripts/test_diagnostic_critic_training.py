"""CPU regression checks; no project models or dataset downloads required."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch

from train_diagnostic_critic import (
    PRECISION_PROTOCOL, constant_baseline, evaluate, forward_value_loss,
    forward_values, load_checkpoint, make_loader, predict_values,
    save_checkpoint, validate_resume_manifest,
)


class TinyCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.value_head = torch.nn.Linear(4, 1)
        self._diagnostic_amp_bf16 = True

    def forward(self, input_ids, **kwargs):
        return self.value_head(self.embedding(input_ids)).squeeze(-1)


class TokenizerStub:
    pad_token_id = 0

    def save_pretrained(self, path):
        pass


class CriticTrainingTests(unittest.TestCase):
    def test_small_adam_updates_survive_fp32_storage(self):
        # The same starting BF16-representable value and 1e-6 update.
        start = torch.tensor([0.02], dtype=torch.bfloat16)
        low = torch.nn.Parameter(start.clone())
        full = torch.nn.Parameter(start.float())
        for parameter in (low, full):
            optimizer = torch.optim.AdamW([parameter], lr=1e-6, weight_decay=0., foreach=False)
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
        self.assertTrue(torch.equal(low.detach(), start))
        self.assertLess(full.item(), start.float().item())

    def test_cache_loss_and_evaluation_share_precision_and_token_mask(self):
        torch.manual_seed(42)
        model = TinyCritic()
        examples = [
            dict(input_ids=[1, 2, 3, 4], prompt_length=2, response_length=2,
                 reward=1., source_index=1, rollout_index=0),
            dict(input_ids=[2, 3], prompt_length=1, response_length=1,
                 reward=-2., source_index=2, rollout_index=0),
        ]
        loader = make_loader(examples, TokenizerStub(), 2, False, 42)
        ids, mask, positions, targets, keys = next(iter(loader))
        self.assertEqual(positions, [[1, 2], [0]])
        cache = predict_values(model, loader, "cpu")
        expected_mse = torch.cat([
            (cache[key] - target).square() for key, target in zip(keys, targets)
        ]).mean().item()
        loss = forward_value_loss(model, ids, mask, positions, targets,
                                  [cache[k] for k in keys], "cpu", .2)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertAlmostEqual(loss.item() * 2, expected_mse, places=6)
        metrics = evaluate(model, loader, "cpu")
        self.assertEqual(metrics["prefix_count"], 3)
        self.assertAlmostEqual(metrics["mse"], expected_mse, places=6)
        loss.backward()
        self.assertTrue(all(p.grad.dtype == torch.float32 for p in model.parameters()))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_checkpoint_preserves_fp32_updates_and_next_optimizer_step(self):
        torch.manual_seed(42)
        model = TinyCritic()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, foreach=False)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)

        def step(net, opt, sched):
            opt.zero_grad()
            values = forward_values(net, torch.tensor([[1, 2]]), torch.ones(1, 2), "cpu")
            values.float().square().mean().backward()
            opt.step()
            sched.step()

        step(model, optimizer, scheduler)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_checkpoint(path, model, optimizer, scheduler, 1, 0, 1., [],
                            TokenizerStub(), 1, 1)
            restored = TinyCritic()
            opt2 = torch.optim.AdamW(restored.parameters(), lr=1e-6, foreach=False)
            sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda step: 1.)
            state = load_checkpoint(path, restored, opt2, sched2, TokenizerStub())
            self.assertEqual(state["round"], 1)
            for values in opt2.state.values():
                self.assertEqual(values["exp_avg"].dtype, torch.float32)
                self.assertEqual(values["exp_avg_sq"].dtype, torch.float32)
            step(model, optimizer, scheduler)
            step(restored, opt2, sched2)
            for a, b in zip(model.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(a, b))

    def test_baseline_uses_training_mean_and_resume_rejects_legacy(self):
        examples = [dict(reward=1., response_length=2), dict(reward=-2., response_length=1)]
        self.assertAlmostEqual(constant_baseline(examples, 0.5), 2.25)
        current = dict(precision_protocol=PRECISION_PROTOCOL, max_rounds=100)
        validate_resume_manifest(current, copy.deepcopy(current))
        with self.assertRaisesRegex(ValueError, "precision_protocol"):
            validate_resume_manifest({}, current)
        with self.assertRaisesRegex(ValueError, "max_rounds"):
            validate_resume_manifest({**current, "max_rounds": 200}, current)


if __name__ == "__main__":
    unittest.main()
