"""Deterministic conditional EDM sampling and stage evaluation."""
import argparse
import gc
import os
import time
from pathlib import Path

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import k_diffusion as K

from .data import PairedRGBDataset, paired_model_transform
from .evaluate_teacher_weights import validate_checkpoint
from .image_metrics import mse_per_sample, psnr_per_sample, ssim_per_sample
from .train_teacher import build_model, save_preview, validate_config, write_json
from .train_teacher_short import load_json, tensor_to_uint8


def validate_sampling_config(config):
    if config["status"] != "experimental_sampling_not_reported_by_thesis":
        raise ValueError("undocumented sampling settings must remain experimental")
    if config["solver"] != "heun" or config["stochastic_churn"] != 0:
        raise ValueError("the first baseline requires deterministic Heun sampling")
    if not 0 < config["sigma_minimum"] < config["sigma_maximum"]:
        raise ValueError("invalid sampling sigma interval")
    if config["rho"] <= 0:
        raise ValueError("rho must be positive")
    steps = config["candidate_steps"]
    if not steps or steps != sorted(set(steps)) or any(step < 2 for step in steps):
        raise ValueError("candidate steps must be unique, increasing and at least two")
    if config["batch_size"] < 1 or config["num_workers"] < 0:
        raise ValueError("invalid data loader settings")
    if not 0 < config["preview_samples"] <= config["pilot_samples"]:
        raise ValueError("invalid pilot or preview sample count")
    if config["weights"] != ["online", "ema"]:
        raise ValueError("the baseline must compare online and EMA weights")


def karras_schedule(steps, config, device):
    return K.sampling.get_sigmas_karras(
        steps,
        config["sigma_minimum"],
        config["sigma_maximum"],
        rho=config["rho"],
        device=device,
    )


@torch.no_grad()
def sample_conditional_edm(model, cloudy, initial_noise, steps, config, *, callback=None):
    if cloudy.shape != initial_noise.shape or cloudy.ndim != 4 or cloudy.shape[1] != 3:
        raise ValueError("cloudy input and initial noise must be matching BCHW RGB tensors")
    sigmas = karras_schedule(steps, config, cloudy.device)
    state = initial_noise * sigmas[0]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cloudy.is_cuda):
        sample = K.sampling.sample_heun(
            model,
            state,
            sigmas,
            extra_args={"cloudy": cloudy},
            disable=True,
            s_churn=config["stochastic_churn"],
            callback=callback,
        )
    if not torch.isfinite(sample).all():
        raise FloatingPointError("sampler produced a nonfinite output")
    return sample.float()


def select_indices(dataset_size, sample_count, seed):
    if sample_count is None:
        return list(range(dataset_size))
    if not 0 < sample_count <= dataset_size:
        raise ValueError("sample count exceeds the test set")
    generator = torch.Generator().manual_seed(seed + 600_000)
    return torch.randperm(dataset_size, generator=generator)[:sample_count].tolist()


@torch.no_grad()
def evaluate_sampling(model, dataset, indices, steps, config, weights_name):
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    noise_generator = torch.Generator(device="cpu").manual_seed(config["seed"] + 700_000)
    totals = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0}
    seen = 0
    preview = None
    torch.cuda.synchronize()
    started = time.monotonic()
    for batch in loader:
        clean = batch["clean"].to("cuda", non_blocking=True)
        cloudy = batch["cloudy"].to("cuda", non_blocking=True)
        initial_noise = torch.randn(clean.shape, generator=noise_generator).to(
            "cuda", non_blocking=True
        )
        prediction = sample_conditional_edm(model, cloudy, initial_noise, steps, config)
        prediction = prediction.clamp(-1, 1)
        clean_float = clean.float()
        values = {
            "mse": mse_per_sample(prediction, clean_float),
            "psnr": psnr_per_sample(prediction, clean_float),
            "ssim": ssim_per_sample(prediction, clean_float),
        }
        for name, per_sample in values.items():
            totals[name] += per_sample.sum().item()
        if preview is None:
            count = min(config["preview_samples"], clean.shape[0])
            preview = {
                "cloudy": [tensor_to_uint8(cloudy[i]) for i in range(count)],
                "prediction": [tensor_to_uint8(prediction[i]) for i in range(count)],
                "clean": [tensor_to_uint8(clean[i]) for i in range(count)],
            }
        seen += clean.shape[0]
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    return (
        {
            "weights": weights_name,
            "steps": steps,
            "function_evaluations": 2 * steps - 1,
            "samples": seen,
            **{name: total / seen for name, total in totals.items()},
            "sampling_seconds": elapsed,
            "seconds_per_image": elapsed / seen,
        },
        preview,
    )


def comparison_rows(online_preview, ema_preview):
    rows = []
    count = len(online_preview["prediction"])
    for index in range(count):
        for key in ("cloudy", "clean"):
            if not np.array_equal(online_preview[key][index], ema_preview[key][index]):
                raise ValueError("online and EMA sampling used different test inputs")
        rows.append(
            np.concatenate(
                [
                    online_preview["cloudy"][index],
                    online_preview["prediction"][index],
                    ema_preview["prediction"][index],
                    online_preview["clean"][index],
                ],
                axis=1,
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=("CUHK-CR1", "CUHK-CR2"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-epoch", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--steps", type=int, nargs="+")
    parser.add_argument("--training-config", type=Path, default=Path("configs/teacher_training.json"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    training_config = load_json(root / args.training_config)
    sampling_config = load_json(root / "configs/teacher_sampling.json")
    validate_config(training_config)
    validate_sampling_config(sampling_config)
    requested_steps = args.steps or sampling_config["candidate_steps"]
    if any(step not in sampling_config["candidate_steps"] for step in requested_steps):
        raise ValueError("requested steps are outside the reviewed candidate set")

    protocol = load_json(root / "configs/data_protocol.json")
    subset_config = protocol["subsets"][args.subset]
    dataset = PairedRGBDataset(
        root / subset_config["test_manifest"],
        pair_transform=paired_model_transform,
        image_size=training_config["image_size"],
    )
    if len(dataset) != subset_config["test_count"]:
        raise AssertionError("test dataset count differs from fixed protocol")
    indices = select_indices(len(dataset), args.sample_count, sampling_config["seed"])

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    validate_checkpoint(checkpoint, args.subset, training_config, args.expected_epoch)
    states = {name: checkpoint.pop("model" if name == "online" else "ema_model")
              for name in sampling_config["weights"]}
    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]
    del checkpoint
    gc.collect()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(root, training_config)
    torch.cuda.reset_peak_memory_stats()
    metrics = []
    previews = {}
    for weights_name in sampling_config["weights"]:
        model.load_state_dict(states[weights_name])
        model.eval()
        for steps in requested_steps:
            record, preview = evaluate_sampling(
                model, dataset, indices, steps, sampling_config, weights_name
            )
            metrics.append(record)
            previews[(weights_name, steps)] = preview
            print(record, flush=True)
    del states
    gc.collect()

    for steps in requested_steps:
        rows = comparison_rows(previews[("online", steps)], previews[("ema", steps)])
        save_preview(args.output_dir / f"online_vs_ema_steps_{steps:03d}.png", rows)
    result = {
        "status": "completed",
        "scope": "full_conditional_edm_sampling",
        "metric_protocol": "ssim_nonnegative_v2_reflect11_sigma1.5_rgb_mean",
        "sampling_status": sampling_config["status"],
        "subset": args.subset,
        "checkpoint": str(checkpoint_path),
        "epoch": epoch,
        "global_step": global_step,
        "test_split_size": len(dataset),
        "evaluated_samples": len(indices),
        "indices": indices,
        "preview_columns": ["cloudy", "online", "ema", "clean"],
        "config": sampling_config,
        "metrics": metrics,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    write_json(args.output_dir / "sampling_results.json", result)
    print({"summary": result}, flush=True)


if __name__ == "__main__":
    main()
