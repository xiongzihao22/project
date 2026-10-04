"""Thesis Section 3.2 operations with unreported choices supplied by callers.

Denoisers have signature model(noisy_rgb, sigma_batch, cloudy_rgb).
Trajectory integration, loss reduction, weighting and EMA decay are explicit.
No optimizer, backbone architecture or noise schedule is silently selected.
"""
import copy
import math

import torch
from torch import nn


def noise_scale(sigma, image, minimum=0.0):
    if image.ndim != 4 or not image.is_floating_point():
        raise ValueError('Expected floating BCHW tensor')
    # Keep coefficient arithmetic in float32 for low precision image tensors.
    dtype = torch.float64 if image.dtype == torch.float64 else torch.float32
    sigma = torch.as_tensor(sigma, device=image.device, dtype=dtype)
    if sigma.ndim == 0:
        sigma = sigma.expand(image.shape[0])
    if sigma.shape != (image.shape[0],):
        raise ValueError('sigma must be scalar or shape [batch]')
    if not torch.isfinite(sigma).all() or (sigma < minimum).any():
        raise ValueError('Invalid noise scale')
    return sigma, sigma[:, None, None, None]


def matching_images(first, second):
    if first.shape != second.shape or first.device != second.device:
        raise ValueError('Images must have matching shape and device')
    if first.ndim != 4 or first.shape[1] != 3:
        raise ValueError('Expected BCHW RGB images')


def add_noise(clean, sigma, noise):
    """Equation (3-1): add noise to the clear target, not the cloudy condition."""
    matching_images(clean, noise)
    _, scale = noise_scale(sigma, clean)
    return clean + scale * noise


class ConsistencyFunction(nn.Module):
    """Equations (3-7)/(3-8); backbone owns concatenation and noise embedding."""
    def __init__(self, backbone, *, sigma_min=0.02, sigma_data=0.5):
        super().__init__()
        if not all(math.isfinite(v) and v > 0 for v in (sigma_min, sigma_data)):
            raise ValueError('Noise constants must be finite and positive')
        self.backbone = backbone
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data

    def forward(self, noisy, sigma, cloudy):
        matching_images(noisy, cloudy)
        sigmas, scale = noise_scale(sigma, noisy, self.sigma_min)
        prediction = self.backbone(noisy, sigmas, cloudy)
        matching_images(noisy, prediction)
        delta = scale - self.sigma_min
        skip = self.sigma_data**2 / (delta.square() + self.sigma_data**2)
        out = self.sigma_data * delta / (self.sigma_data**2 + scale.square()).sqrt()
        return skip * noisy + out * prediction


def squared_error_per_sample(prediction, target, reduction):
    matching_images(prediction, target)
    errors = (prediction - target).square().flatten(1)
    if reduction == 'pixel_mean':
        return errors.mean(1)
    if reduction == 'squared_l2':
        return errors.sum(1)
    raise ValueError('Explicitly choose pixel_mean or squared_l2')


def teacher_loss(teacher, clean, cloudy, sigma, noise, *, weights, reduction,
                 reconstruction_l1_weight=0.0, return_components=False, return_per_sample=False):
    """Equation (3-3); the caller supplies w(sigma) and the teacher wrapper."""
    if (isinstance(reconstruction_l1_weight, bool)
            or not math.isfinite(reconstruction_l1_weight) or reconstruction_l1_weight < 0):
        raise ValueError('Invalid teacher reconstruction weight')
    matching_images(clean, cloudy)
    sigmas, _ = noise_scale(sigma, clean)
    predicted = teacher(add_noise(clean, sigmas, noise), sigmas, cloudy)
    per_sample = squared_error_per_sample(predicted, clean, reduction)
    weights = torch.as_tensor(weights, device=clean.device, dtype=per_sample.dtype)
    if weights.shape != per_sample.shape:
        raise ValueError('weights must have shape [batch]')
    if not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Teacher weights must be finite and positive')
    edm = (weights * per_sample).mean()
    if reconstruction_l1_weight == 0 and not return_components:
        return edm
    reconstruction = (predicted.float() - clean.float()).abs().mean()
    loss = edm if reconstruction_l1_weight == 0 else edm + reconstruction_l1_weight * reconstruction
    if return_components:
        components = {'edm_loss': edm.detach(), 'reconstruction_l1': reconstruction.detach()}
        if return_per_sample:
            components['denoising_mse_per_sample'] = per_sample.detach()
        return loss, components
    return loss


def make_target(student):
    target = copy.deepcopy(student)
    target.requires_grad_(False)
    target.eval()
    return target


@torch.no_grad()
def update_ema(target, student, *, decay):
    if not math.isfinite(decay) or not 0 <= decay < 1:
        raise ValueError('EMA decay must be in [0, 1)')
    target_params = dict(target.named_parameters())
    online_params = dict(student.named_parameters())
    if target_params.keys() != online_params.keys():
        raise ValueError('EMA models must have identical parameter names')
    validate_fixed_buffers(target, student)
    for name in target_params:
        if target_params[name].shape != online_params[name].shape:
            raise ValueError('EMA parameter shapes differ')
    for name, parameter in target_params.items():
        parameter.lerp_(online_params[name], 1 - decay)


def validate_fixed_buffers(target, student):
    """HDiT Fourier/RoPE buffers are copied at creation and must stay fixed."""
    target_buffers = dict(target.named_buffers())
    online_buffers = dict(student.named_buffers())
    if target_buffers.keys() != online_buffers.keys():
        raise ValueError('EMA buffer names differ')
    for name, value in target_buffers.items():
        other = online_buffers[name]
        if (value.shape != other.shape or value.dtype != other.dtype
                or value.device != other.device or not torch.equal(value, other)):
            raise ValueError('Fixed EMA buffer differs: ' + name)


def distillation_loss(student, target, teacher, clean, cloudy, sigma_high,
                      sigma_low, noise, *, trajectory_step, reduction,
                      reconstruction_weight=0.0, return_components=False):
    """Equation (3-5). trajectory_step(teacher, x, high, low, y) returns x_low."""
    matching_images(clean, cloudy)
    if (isinstance(reconstruction_weight, bool) or not math.isfinite(reconstruction_weight)
            or reconstruction_weight < 0):
        raise ValueError('Invalid reconstruction weight')
    high, _ = noise_scale(sigma_high, clean)
    low, _ = noise_scale(sigma_low, clean)
    if (low <= 0).any() or (high <= low).any():
        raise ValueError('Require sigma_high > sigma_low > 0')
    if teacher.training or target.training:
        raise ValueError('Teacher and target must be in evaluation mode')
    if any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('Teacher must be frozen before distillation')
    if any(p.requires_grad for p in target.parameters()):
        raise ValueError('Target must only be updated by EMA')
    x_high = add_noise(clean, high, noise)
    with torch.no_grad():
        x_low = trajectory_step(teacher, x_high, high, low, cloudy)
        matching_images(x_high, x_low)
        target_output = target(x_low, low, cloudy)
    online_output = student(x_high, high, cloudy)
    consistency = squared_error_per_sample(online_output, target_output.detach(), reduction).mean()
    reconstruction = (online_output.float() - clean.float()).abs().mean()
    loss = consistency if reconstruction_weight == 0 else consistency + reconstruction_weight * reconstruction
    if return_components:
        return loss, {"consistency_loss": consistency.detach(),
                      "reconstruction_l1": reconstruction.detach()}
    return loss


def student_step(student, target, teacher, optimizer, batch, *, trajectory_step,
                 reduction, ema_decay):
    """One update; batch supplies clean/cloudy/sigma_high/sigma_low/noise."""
    if not math.isfinite(ema_decay) or not 0 <= ema_decay < 1:
        raise ValueError('Invalid EMA decay')
    validate_fixed_buffers(target, student)
    student.train()
    optimizer.zero_grad(set_to_none=True)
    loss = distillation_loss(student, target, teacher, **batch,
                             trajectory_step=trajectory_step, reduction=reduction)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite distillation loss')
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all()
           for p in student.parameters()):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError('Nonfinite gradient; optimizer not advanced')
    optimizer.step()
    update_ema(target, student, decay=ema_decay)
    return loss.detach().item()
