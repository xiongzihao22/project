"""CPU training unit tests with synthetic tensors."""
import unittest

import torch
from torch import nn

from cloud_removal.training import (ConsistencyFunction, add_noise, make_target,
    teacher_loss, distillation_loss, student_step, update_ema)


class TestBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(6, 3, 1)

    def forward(self, noisy, sigma, cloudy):
        return self.projection(torch.cat([noisy, cloudy], dim=1))


def test_trajectory(teacher, noisy, high, low, cloudy):
    # Synthetic ODE solver fixture.
    return noisy + ((low - high) / high)[:, None, None, None] * (
        noisy - teacher(noisy, high, cloudy))


class TrainingTests(unittest.TestCase):
    def test_reconstruction_adds_exact_l1_and_preserves_zero_weight(self):
        kwargs = dict(trajectory_step=test_trajectory, reduction='pixel_mean')
        baseline = distillation_loss(self.student, self.target, self.teacher, **self.batch, **kwargs)
        zero, parts = distillation_loss(self.student, self.target, self.teacher, **self.batch,
                                       **kwargs, return_components=True)
        self.assertTrue(torch.equal(baseline, zero))
        weighted, parts = distillation_loss(self.student, self.target, self.teacher, **self.batch,
            **kwargs, reconstruction_weight=0.0001, return_components=True)
        output = self.student(add_noise(self.batch['clean'], self.batch['sigma_high'], self.batch['noise']),
                              self.batch['sigma_high'], self.batch['cloudy'])
        expected = (output.float() - self.batch['clean'].float()).abs().mean()
        self.assertTrue(torch.allclose(parts['reconstruction_l1'], expected))
        self.assertTrue(torch.allclose(weighted, baseline + 0.0001 * expected))
        weighted.backward()
        self.assertTrue(any(p.grad is not None for p in self.student.parameters()))
        self.assertTrue(all(p.grad is None for p in self.teacher.parameters()))
        self.assertTrue(all(p.grad is None for p in self.target.parameters()))

    def setUp(self):
        torch.manual_seed(13)
        self.student = ConsistencyFunction(TestBackbone())
        self.target = make_target(self.student)
        self.teacher = TestBackbone().eval().requires_grad_(False)
        self.batch = dict(clean=torch.randn(2, 3, 8, 8),
                          cloudy=torch.randn(2, 3, 8, 8),
                          noise=torch.randn(2, 3, 8, 8),
                          sigma_high=torch.tensor([1., 2.]),
                          sigma_low=torch.tensor([0.1, 0.2]))

    def test_tensor_boundary(self):
        x = self.batch['clean']
        self.assertTrue(torch.equal(self.student(x, 0.02, self.batch['cloudy']), x))

    def test_noise_formula(self):
        x = self.batch['clean']
        n = self.batch['noise']
        self.assertTrue(torch.equal(add_noise(x, 2., n), x + 2 * n))

    def test_noise_shape_rejected(self):
        with self.assertRaises(ValueError):
            add_noise(self.batch['clean'], torch.ones(2, 1), self.batch['noise'])

    def test_condition_changes_output(self):
        x = self.batch['clean']
        a = self.student(x, 1., torch.zeros_like(x))
        b = self.student(x, 1., torch.ones_like(x))
        self.assertFalse(torch.equal(a, b))

    def test_teacher_loss_backpropagates(self):
        model = TestBackbone()
        loss = teacher_loss(model, self.batch['clean'], self.batch['cloudy'],
                            self.batch['sigma_high'], self.batch['noise'],
                            weights=torch.ones(2), reduction='pixel_mean')
        loss.backward()
        self.assertIsNotNone(model.projection.weight.grad)

    def test_student_and_ema_update_but_teacher_frozen(self):
        before = {k: p.detach().clone() for k, p in self.student.named_parameters()}
        teacher_before = {k: p.detach().clone() for k, p in self.teacher.named_parameters()}
        # Test-specific optimizer and EMA settings.
        optimizer = torch.optim.SGD(self.student.parameters(), lr=0.01)
        value = student_step(self.student, self.target, self.teacher, optimizer,
                             self.batch, trajectory_step=test_trajectory,
                             reduction='pixel_mean', ema_decay=0.9)
        self.assertGreaterEqual(value, 0)
        self.assertTrue(any(not torch.equal(before[k], p) for k, p in self.student.named_parameters()))
        for k, p in self.target.named_parameters():
            expected = before[k] * 0.9 + dict(self.student.named_parameters())[k] * 0.1
            self.assertTrue(torch.allclose(p, expected))
            self.assertIsNone(p.grad)
        for k, p in self.teacher.named_parameters():
            self.assertTrue(torch.equal(p, teacher_before[k]))
            self.assertIsNone(p.grad)

    def test_wrong_noise_order_rejected(self):
        self.batch['sigma_low'] = torch.tensor([3., 3.])
        with self.assertRaises(ValueError):
            distillation_loss(self.student, self.target, self.teacher, **self.batch,
                              trajectory_step=test_trajectory, reduction='pixel_mean')

    def test_trainable_teacher_rejected(self):
        self.teacher.requires_grad_(True)
        with self.assertRaises(ValueError):
            distillation_loss(self.student, self.target, self.teacher, **self.batch,
                              trajectory_step=test_trajectory, reduction='pixel_mean')

    def test_ema_invalid_decay_rejected(self):
        with self.assertRaises(ValueError):
            update_ema(self.target, self.student, decay=1.1)


if __name__ == '__main__':
    unittest.main()
