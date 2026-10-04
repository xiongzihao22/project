"""Approved conditional EDM teacher; no training schedule is selected here."""
import math

from torch import nn

from .training import matching_images, noise_scale


def identity_inputs(noisy, sigma, cloudy):
    return noisy, cloudy


class EDMInputScaling:
    def __init__(self, sigma_data=0.5):
        if not math.isfinite(sigma_data) or sigma_data <= 0:
            raise ValueError('sigma_data must be finite and positive')
        self.sigma_data = sigma_data

    def __call__(self, noisy, sigma, cloudy):
        _, scale = noise_scale(sigma, noisy)
        return noisy / (scale.square() + self.sigma_data**2).sqrt(), cloudy


class EDMTeacher(nn.Module):
    """Backbone must already scale its noisy input exactly once."""
    def __init__(self, backbone, sigma_data=0.5):
        super().__init__()
        if not math.isfinite(sigma_data) or sigma_data <= 0:
            raise ValueError('sigma_data must be finite and positive')
        self.backbone = backbone
        self.sigma_data = sigma_data

    def forward(self, noisy, sigma, cloudy):
        matching_images(noisy, cloudy)
        sigmas, scale = noise_scale(sigma, noisy)
        prediction = self.backbone(noisy, sigmas, cloudy)
        matching_images(noisy, prediction)
        denominator = scale.square() + self.sigma_data**2
        skip = self.sigma_data**2 / denominator
        out = scale * self.sigma_data / denominator.sqrt()
        return skip * noisy + out * prediction
