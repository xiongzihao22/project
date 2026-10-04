"""Synthetic checks for adjacent-state student trajectories."""
import unittest

import torch
from torch import nn

from cloud_removal.student_trajectory import (
    karras_noise_levels,
    sample_adjacent_sigmas,
    teacher_pf_ode_step,
)


class ZeroDenoiser(nn.Module):
    def forward(self, noisy, sigma, cloudy):
        return torch.zeros_like(noisy)


class PerfectDenoiser(nn.Module):
    def forward(self, noisy, sigma, cloudy):
        return noisy


class ConditionalOracle(nn.Module):
    def forward(self, noisy, sigma, cloudy):
        return cloudy


class StudentTrajectoryTests(unittest.TestCase):
    def test_conditional_oracle_preserves_known_noise_path(self):
        generator = torch.Generator().manual_seed(42)
        clean = torch.randn(3, 3, 8, 8, generator=generator)
        noise = torch.randn(3, 3, 8, 8, generator=generator)
        high = torch.tensor([20.0, 3.0, 0.2])
        low = torch.tensor([8.0, 0.5, 0.02])
        noisy = clean + high[:, None, None, None] * noise
        expected = clean + low[:, None, None, None] * noise
        # The condition is the oracle's clean image only in this analytic fixture.
        for solver in ("euler", "heun"):
            with self.subTest(solver=solver):
                actual = teacher_pf_ode_step(
                    ConditionalOracle(), noisy, high, low, clean, solver=solver
                )
                torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-6)

    def test_trajectory_output_is_detached(self):
        noisy = torch.randn(2, 3, 4, 4, requires_grad=True)
        cloudy = torch.randn_like(noisy, requires_grad=True)
        for solver in ("euler", "heun"):
            output = teacher_pf_ode_step(
                ConditionalOracle(), noisy, 2.0, 0.5, cloudy, solver=solver
            )
            self.assertFalse(output.requires_grad)
            self.assertIsNone(output.grad_fn)

    def test_grid_and_adjacent_sampling(self):
        levels = karras_noise_levels(18, 0.02, 20, 7, device="cpu")
        self.assertEqual(len(levels), 18)
        self.assertAlmostEqual(levels[0].item(), 20, places=4)
        self.assertAlmostEqual(levels[-1].item(), 0.02, places=5)
        self.assertTrue(torch.equal(levels[-1], torch.tensor(0.02)))
        high, low = sample_adjacent_sigmas(
            levels, 100, generator=torch.Generator().manual_seed(13)
        )
        self.assertTrue(torch.all(high > low))

    def test_euler_zero_denoiser_follows_known_solution(self):
        noisy = torch.ones(2, 3, 4, 4)
        cloudy = torch.zeros_like(noisy)
        low = teacher_pf_ode_step(
            ZeroDenoiser(), noisy, torch.tensor([2.0, 1.0]),
            torch.tensor([1.0, 0.5]), cloudy, solver="euler"
        )
        self.assertTrue(torch.allclose(low, noisy * 0.5))

    def test_heun_and_euler_preserve_constant_path(self):
        noisy = torch.randn(2, 3, 4, 4)
        cloudy = torch.randn_like(noisy)
        for solver in ("euler", "heun"):
            low = teacher_pf_ode_step(
                PerfectDenoiser(), noisy, 2.0, 0.5, cloudy, solver=solver
            )
            self.assertTrue(torch.equal(low, noisy))

    def test_bad_order_and_solver_rejected(self):
        noisy = torch.zeros(1, 3, 4, 4)
        with self.assertRaises(ValueError):
            teacher_pf_ode_step(
                ZeroDenoiser(), noisy, 0.5, 2.0, noisy, solver="euler"
            )
        with self.assertRaises(ValueError):
            teacher_pf_ode_step(
                ZeroDenoiser(), noisy, 2.0, 0.5, noisy, solver="implicit"
            )


if __name__ == "__main__":
    unittest.main()
