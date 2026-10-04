"""Approved experimental supervision at the pure-noise inference endpoint."""
import math

import torch

from .training import matching_images, noise_scale


def validate_endpoint_config(config):
    weight = config.get('endpoint_weight', 0.0)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight not in (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
        raise ValueError('unapproved endpoint weight')
    if weight and config['sigma']['maximum'] != 20.0:
        raise ValueError('approved endpoint sigma must be 20')


def normalize_endpoint_transition(saved, requested, enabled):
    validate_endpoint_config(saved)
    validate_endpoint_config(requested)
    before, after = saved.get('endpoint_weight', 0.0), requested.get('endpoint_weight', 0.0)
    approved = (before == 0.0 and after in (1e-4, 3e-4, 1e-3)) or (before, after) in ((1e-3, 3e-3), (3e-3, 1e-2))
    if before != after and not (enabled and approved):
        raise ValueError('endpoint loss change is not authorized')
    if enabled:
        expected = {**saved, 'maximum_epochs': requested['maximum_epochs'], 'endpoint_weight': after}
        if requested != expected or requested['maximum_epochs'] < saved['maximum_epochs']:
            raise ValueError('endpoint transition cannot change other settings')
    result = dict(requested)
    if 'endpoint_weight' in saved:
        result['endpoint_weight'] = before
    else:
        result.pop('endpoint_weight', None)
    return result


def endpoint_l1(student, clean, cloudy, noise, sigma_max):
    matching_images(clean, cloudy)
    matching_images(clean, noise)
    if not math.isfinite(sigma_max) or sigma_max != 20.0:
        raise ValueError('approved endpoint sigma must be 20')
    sigmas, scale = noise_scale(sigma_max, clean)
    # Reuse the paired Gaussian draw, without clean-image content. Preserve RNG
    # around this extra forward so it cannot shift the next consistency draw.
    devices = [clean.device.index] if clean.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        prediction = student(noise.detach() * scale, sigmas, cloudy)
    matching_images(prediction, clean)
    return (prediction.float() - clean.detach().float()).abs().mean()
