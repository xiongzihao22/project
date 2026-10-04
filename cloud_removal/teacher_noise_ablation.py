"""Approved high-noise mixture with a separate, checkpointable CPU RNG."""
import copy
import math

import torch


def validate_mixture(config):
    mixture = config.get("high_noise_mixture")
    if mixture is None:
        return
    if set(mixture) != {"probability", "minimum", "maximum"}:
        raise ValueError("unexpected high-noise mixture fields")
    p, low, high = (mixture[k] for k in ("probability", "minimum", "maximum"))
    if not all(math.isfinite(v) for v in (p, low, high)):
        raise ValueError("mixture settings must be finite")
    if not 0 < p <= 1 or not config["minimum"] <= low < high <= config["maximum"]:
        raise ValueError("invalid mixture probability or bounds")


def replace_with_high_noise(base, config, generator):
    validate_mixture(config)
    mixture = config.get("high_noise_mixture")
    if mixture is None:
        return base
    if generator is None or generator.device.type != "cpu":
        raise ValueError("mixture requires an independent CPU generator")
    mask = torch.rand(base.shape, generator=generator) < mixture["probability"]
    uniform = torch.rand(base.shape, generator=generator)
    high = (math.log(mixture["minimum"]) + uniform * (
        math.log(mixture["maximum"]) - math.log(mixture["minimum"])
    )).exp().clamp(mixture["minimum"], mixture["maximum"])
    return torch.where(mask.to(base.device), high.to(base.device), base)


def validate_noise_transition(saved, current):
    expected = copy.deepcopy(saved)
    if "high_noise_mixture" in expected["sigma"]:
        raise ValueError("only adding a mixture to the original baseline is permitted")
    if "high_noise_mixture" not in current["sigma"]:
        raise ValueError("no mixture transition requested")
    validate_mixture(current["sigma"])
    expected["sigma"]["high_noise_mixture"] = current["sigma"]["high_noise_mixture"]
    if expected != current:
        raise ValueError("noise transition must not change any other setting")
