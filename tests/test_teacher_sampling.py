import unittest

import torch
from torch import nn

from cloud_removal.teacher_sampling import (
    karras_schedule,
    sample_conditional_edm,
    select_indices,
    validate_sampling_config,
)


def config():
    return {
        "status": "experimental_sampling_not_reported_by_thesis",
        "solver": "heun",
        "sigma_minimum": 0.02,
        "sigma_maximum": 20.0,
        "rho": 7.0,
        "candidate_steps": [18, 32, 40],
        "stochastic_churn": 0.0,
        "seed": 13,
        "batch_size": 4,
        "num_workers": 4,
        "pilot_samples": 8,
        "preview_samples": 2,
        "weights": ["online", "ema"],
    }


class ZeroDenoiser(nn.Module):
    def forward(self, noisy, sigma, cloudy):
        return torch.zeros_like(noisy)


class TeacherSamplingTests(unittest.TestCase):
    def test_read_only_callback_preserves_heun_output(self):
        class Denoiser(nn.Module):
            def forward(self, noisy, sigma, cloudy):
                return noisy * 0.25 + cloudy * 0.5
        cloudy = torch.randn(2, 3, 4, 4)
        noise = torch.randn_like(cloudy)
        observations = []
        expected = sample_conditional_edm(Denoiser(), cloudy, noise, 32, config())
        actual = sample_conditional_edm(
            Denoiser(), cloudy, noise, 32, config(),
            callback=lambda state: observations.append((state["i"], state["sigma"].item())),
        )
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual([i for i, _ in observations], list(range(32)))
        self.assertGreater(observations[0][1], observations[-1][1])

    def test_reviewed_sampling_config(self):
        validate_sampling_config(config())

    def test_karras_schedule_descends_and_ends_at_zero(self):
        sigmas = karras_schedule(18, config(), "cpu")
        self.assertEqual(len(sigmas), 19)
        self.assertAlmostEqual(sigmas[0].item(), 20.0, places=5)
        self.assertAlmostEqual(sigmas[-2].item(), 0.02, places=5)
        self.assertEqual(sigmas[-1].item(), 0.0)
        self.assertTrue((sigmas[:-1] > sigmas[1:]).all())

    def test_zero_denoiser_reaches_zero(self):
        cloudy = torch.ones(2, 3, 4, 4)
        noise = torch.randn_like(cloudy)
        sample = sample_conditional_edm(ZeroDenoiser(), cloudy, noise, 18, config())
        torch.testing.assert_close(sample, torch.zeros_like(sample), atol=1e-5, rtol=0)

    def test_fixed_indices(self):
        first = select_indices(20, 8, 13)
        second = select_indices(20, 8, 13)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 8)


if __name__ == "__main__":
    unittest.main()
