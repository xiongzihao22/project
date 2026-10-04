"""Explicit noise grid and conditional teacher PF-ODE steps for distillation."""
import math

import torch

from .training import matching_images, noise_scale


def karras_noise_levels(count, sigma_min, sigma_max, rho, *, device):
    if not isinstance(count, int) or count < 2:
        raise ValueError("noise grid needs at least two levels")
    if not all(math.isfinite(value) for value in (sigma_min, sigma_max, rho)):
        raise ValueError("noise grid values must be finite")
    if not 0 < sigma_min < sigma_max or rho <= 0:
        raise ValueError("invalid noise grid bounds or rho")
    ramp = torch.linspace(0, 1, count, dtype=torch.float32, device=device)
    inverse_rho = 1 / rho
    levels = (
        sigma_max**inverse_rho
        + ramp * (sigma_min**inverse_rho - sigma_max**inverse_rho)
    ).pow(rho)
    levels[0] = sigma_max
    levels[-1] = sigma_min
    return levels


def sample_adjacent_sigmas(levels, batch_size, *, generator, high_noise_probability=0.0):
    if levels.ndim != 1 or len(levels) < 2:
        raise ValueError("expected a one-dimensional grid")
    if not torch.isfinite(levels).all() or not torch.all(levels[:-1] > levels[1:]):
        raise ValueError("noise levels must be finite and strictly descending")
    if levels[-1] <= 0 or batch_size < 1:
        raise ValueError("noise levels and batch size must be positive")
    if high_noise_probability not in (0.0, 0.5):
        raise ValueError("unapproved high-noise probability")
    if high_noise_probability and len(levels) != 18:
        raise ValueError("high-noise mixture requires the approved 18-level grid")
    index = torch.randint(0, len(levels) - 1, (batch_size,), generator=generator)
    if high_noise_probability:
        high_index = torch.randint(0, 4, (batch_size,), generator=generator)
        choose_high = torch.rand(batch_size, generator=generator) < high_noise_probability
        index = torch.where(choose_high, high_index, index)
    index = index.to(levels.device)
    return levels[index], levels[index + 1]


@torch.no_grad()
def teacher_pf_ode_step(teacher, noisy, sigma_high, sigma_low, cloudy, *, solver):
    matching_images(noisy, cloudy)
    high, high_scale = noise_scale(sigma_high, noisy)
    low, low_scale = noise_scale(sigma_low, noisy)
    if (low <= 0).any() or (high <= low).any():
        raise ValueError("require sigma_high > sigma_low > 0")
    if solver not in ("euler", "heun"):
        raise ValueError("solver must be explicitly euler or heun")
    first = (noisy - teacher(noisy, high, cloudy)) / high_scale
    delta = low_scale - high_scale
    proposal = noisy + delta * first
    if solver == "euler":
        return proposal
    second = (proposal - teacher(proposal, low, cloudy)) / low_scale
    return noisy + delta * (first + second) / 2
