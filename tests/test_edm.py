import unittest

import torch
from torch import nn

from cloud_removal.edm import EDMInputScaling, EDMTeacher
from cloud_removal.training import make_target, update_ema


class ZeroBackbone(nn.Module):
    def forward(self, noisy, sigma, cloudy):
        return torch.zeros_like(noisy)


class EDMTests(unittest.TestCase):
    def test_only_state_scaled(self):
        x = torch.ones(2, 3, 4, 4)
        y = torch.randn_like(x)
        scaled, condition = EDMInputScaling()(x, torch.tensor([0.5, 1.]), y)
        self.assertIs(condition, y)
        expected = x / torch.tensor([0.5, 1.25]).sqrt()[:, None, None, None]
        torch.testing.assert_close(scaled, expected)

    def test_teacher_output_coefficients(self):
        x = torch.ones(2, 3, 4, 4)
        output = EDMTeacher(ZeroBackbone())(x, 0.5, x)
        torch.testing.assert_close(output, x * 0.5)

    def test_fixed_buffers_preserved_and_changes_rejected(self):
        student = nn.Linear(2, 2)
        student.register_buffer('frequency', torch.arange(3.))
        target = make_target(student)
        update_ema(target, student, decay=0.9)
        student.frequency.add_(1)
        before = target.weight.detach().clone()
        with self.assertRaises(ValueError):
            update_ema(target, student, decay=0.9)
        torch.testing.assert_close(before, target.weight)


if __name__ == '__main__':
    unittest.main()
