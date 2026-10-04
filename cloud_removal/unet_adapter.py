"""Explicit conditional convolutional student for the backbone ablation."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .training import matching_images, noise_scale


def build_unet(spec):
    from k_diffusion.models.image_v1 import ImageDenoiserModelV1
    approved = dict(c_in=3, unet_cond_dim=3, feats_in=256,
                    depths=[2, 2, 2, 2], channels=[128, 256, 384, 512],
                    self_attn_depths=[False] * 4, patch_size=2, dropout_rate=0.0)
    if spec != approved:
        raise ValueError('U-Net specification differs from the approved ablation')
    return ImageDenoiserModelV1(**spec)


class ConditionedUNet(nn.Module):
    def __init__(self, unet, *, input_transform, gradient_checkpointing=True):
        super().__init__()
        self.unet = unet
        self.input_transform = input_transform
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, noisy, sigma, cloudy):
        matching_images(noisy, cloudy)
        sigmas, _ = noise_scale(sigma, noisy)
        if (sigmas <= 0).any():
            raise ValueError('U-Net noise embedding requires positive sigma')
        state, condition = self.input_transform(noisy, sigmas, cloudy)
        matching_images(noisy, state)
        matching_images(noisy, condition)

        def denoise(x, s, c):
            return self.unet(x, s, unet_cond=c)

        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            output = checkpoint(denoise, state, sigmas, condition, use_reentrant=False)
        else:
            output = denoise(state, sigmas, condition)
        matching_images(noisy, output)
        return output
