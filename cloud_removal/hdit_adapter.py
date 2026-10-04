"""Section 3.3 RGB conditioning adapter for the official HDiT interface.

The supplied HDiT must accept six input channels and produce three channels.
Its forward(x, sigma) performs the noise embedding internally. Architecture
and input scaling remain explicit; this module does not construct a backbone.
"""
import torch
from torch import nn
from .training import matching_images, noise_scale


class ConditionedHDiT(nn.Module):
    def __init__(self, hdit, *, input_transform):
        super().__init__()
        self.hdit = hdit
        self.input_transform = input_transform

    def forward(self, noisy, sigma, cloudy):
        matching_images(noisy, cloudy)
        sigmas, _ = noise_scale(sigma, noisy)
        if (sigmas <= 0).any():
            raise ValueError('HDiT logarithmic noise embedding requires positive sigma')
        # Transform both streams explicitly; do not silently normalize conditions.
        state, condition = self.input_transform(noisy, sigmas, cloudy)
        matching_images(noisy, state)
        matching_images(noisy, condition)
        output = self.hdit(torch.cat((state, condition), dim=1), sigmas)
        matching_images(noisy, output)
        return output
