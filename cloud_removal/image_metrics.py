"""Image metrics for tensors in the model's [-1, 1] range."""
import math

import torch
from torch.nn import functional as F


def _matching_rgb(prediction, target):
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError("Expected matching BCHW RGB tensors")
    if prediction.device != target.device:
        raise ValueError("Metric tensors must share a device")


def mse_per_sample(prediction, target):
    _matching_rgb(prediction, target)
    return (prediction.float() - target.float()).square().flatten(1).mean(1)


def psnr_per_sample(prediction, target, *, data_range=2.0):
    if not math.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range must be finite and positive")
    mse = mse_per_sample(prediction, target)
    peak = torch.as_tensor(data_range**2, device=mse.device, dtype=mse.dtype)
    return 10 * torch.log10(peak / mse.clamp_min(torch.finfo(mse.dtype).tiny))


def _gaussian_window(size, sigma, *, device, dtype):
    coordinates = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    kernel = torch.exp(-(coordinates.square()) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return (kernel[:, None] * kernel[None, :]).expand(3, 1, size, size).contiguous()


def ssim_per_sample(prediction, target, *, data_range=2.0, window_size=11, sigma=1.5):
    """SSIM after shifting centered model intensities to a nonnegative range."""
    _matching_rgb(prediction, target)
    if not math.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range must be finite and positive")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be positive and odd")
    if min(prediction.shape[-2:]) < window_size:
        raise ValueError("Images are smaller than the SSIM window")
    # SSIM's luminance term is not translation invariant. This is equivalent
    # to evaluating (x + 1) / 2 with data_range=1 for the default model range.
    prediction = prediction.float() + data_range / 2
    target = target.float() + data_range / 2
    window = _gaussian_window(
        window_size, sigma, device=prediction.device, dtype=prediction.dtype
    )
    padding = window_size // 2

    def blur(value):
        value = F.pad(value, (padding, padding, padding, padding), mode="reflect")
        return F.conv2d(value, window, groups=3)

    mean_prediction = blur(prediction)
    mean_target = blur(target)
    prediction_variance = blur(prediction.square()) - mean_prediction.square()
    target_variance = blur(target.square()) - mean_target.square()
    covariance = blur(prediction * target) - mean_prediction * mean_target
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mean_prediction * mean_target + c1) * (2 * covariance + c2)
    denominator = (
        mean_prediction.square() + mean_target.square() + c1
    ) * (prediction_variance + target_variance + c2)
    return (numerator / denominator).flatten(1).mean(1)
