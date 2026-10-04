"""Compare online and EMA teacher weights on the fixed test split."""
import argparse
import gc
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PairedRGBDataset, paired_model_transform
from .image_metrics import mse_per_sample, psnr_per_sample, ssim_per_sample
from .train_teacher import build_model, save_preview, validate_config, write_json
from .train_teacher_short import load_json, tensor_to_uint8


def validate_checkpoint(checkpoint, subset, config, expected_epoch):
    required = {"kind", "subset", "epoch", "global_step", "config", "model", "ema_model"}
    if not isinstance(checkpoint, dict) or not required <= checkpoint.keys():
        raise ValueError("checkpoint lacks the required training state")
    if checkpoint["kind"] != "full_training_state" or checkpoint["subset"] != subset:
        raise ValueError("checkpoint kind or subset mismatch")
    if checkpoint["config"] != config:
        raise ValueError("checkpoint configuration mismatch")
    if checkpoint["epoch"] != expected_epoch:
        raise ValueError("checkpoint epoch differs from the requested stage")


def state_distance(online_state, ema_state):
    if online_state.keys() != ema_state.keys():
        raise ValueError("online and EMA state keys differ")
    online_square_sum = 0.0
    difference_square_sum = 0.0
    maximum_absolute_difference = 0.0
    floating_elements = 0
    for name in online_state:
        online = online_state[name]
        ema = ema_state[name]
        if online.shape != ema.shape or online.dtype != ema.dtype:
            raise ValueError(f"state tensor mismatch: {name}")
        if not online.is_floating_point():
            if not torch.equal(online, ema):
                raise ValueError(f"non-floating state differs: {name}")
            continue
        online_float = online.float()
        difference = online_float - ema.float()
        online_square_sum += online_float.square().sum(dtype=torch.float64).item()
        difference_square_sum += difference.square().sum(dtype=torch.float64).item()
        maximum_absolute_difference = max(
            maximum_absolute_difference, difference.abs().max().item()
        )
        floating_elements += online.numel()
    online_l2 = online_square_sum**0.5
    difference_l2 = difference_square_sum**0.5
    return {
        "floating_elements": floating_elements,
        "online_l2": online_l2,
        "difference_l2": difference_l2,
        "relative_l2": difference_l2 / online_l2 if online_l2 else 0.0,
        "difference_rms": (difference_square_sum / floating_elements) ** 0.5,
        "maximum_absolute_difference": maximum_absolute_difference,
    }


@torch.no_grad()
def evaluate_weights(model, dataset, config, weights_name):
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=False,
        persistent_workers=config["num_workers"] > 0,
    )
    monitor = config["monitor"]
    records = []
    preview = None
    model.eval()
    for sigma_index, sigma_value in enumerate(monitor["sigmas"]):
        totals = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0}
        seen = 0
        noise_generator = torch.Generator(device="cpu").manual_seed(
            config["seed"] + 500_000 + sigma_index
        )
        for batch in loader:
            clean = batch["clean"].to("cuda", non_blocking=True)
            cloudy = batch["cloudy"].to("cuda", non_blocking=True)
            noise = torch.randn(clean.shape, generator=noise_generator).to(
                "cuda", non_blocking=True
            )
            sigmas = torch.full((clean.shape[0],), sigma_value, device="cuda")
            noisy = clean + sigmas[:, None, None, None] * noise
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(noisy, sigmas, cloudy)
            prediction = prediction.float().clamp(-1, 1)
            clean_float = clean.float()
            values = {
                "mse": mse_per_sample(prediction, clean_float),
                "psnr": psnr_per_sample(prediction, clean_float),
                "ssim": ssim_per_sample(prediction, clean_float),
            }
            for name, per_sample in values.items():
                totals[name] += per_sample.sum().item()
            if sigma_value == monitor["preview_sigma"] and preview is None:
                count = min(monitor["preview_count"], clean.shape[0])
                preview = {
                    "cloudy": [tensor_to_uint8(cloudy[i]) for i in range(count)],
                    "noisy": [tensor_to_uint8(noisy[i]) for i in range(count)],
                    "prediction": [tensor_to_uint8(prediction[i]) for i in range(count)],
                    "clean": [tensor_to_uint8(clean[i]) for i in range(count)],
                }
            seen += clean.shape[0]
        records.append(
            {
                "weights": weights_name,
                "sigma": sigma_value,
                "samples": seen,
                **{name: total / seen for name, total in totals.items()},
            }
        )
    if preview is None:
        raise RuntimeError("preview sigma did not produce a preview")
    return records, preview


def metric_deltas(online_records, ema_records):
    if len(online_records) != len(ema_records):
        raise ValueError("metric record counts differ")
    deltas = []
    for online, ema in zip(online_records, ema_records):
        if online["sigma"] != ema["sigma"] or online["samples"] != ema["samples"]:
            raise ValueError("metric record identity differs")
        deltas.append(
            {
                "sigma": online["sigma"],
                "samples": online["samples"],
                "ema_minus_online_mse": ema["mse"] - online["mse"],
                "ema_minus_online_psnr": ema["psnr"] - online["psnr"],
                "ema_minus_online_ssim": ema["ssim"] - online["ssim"],
            }
        )
    return deltas


def comparison_rows(online_preview, ema_preview):
    rows = []
    count = len(online_preview["prediction"])
    for key in ("cloudy", "noisy", "clean"):
        if len(online_preview[key]) != count or len(ema_preview[key]) != count:
            raise ValueError("preview lengths differ")
    for index in range(count):
        for key in ("cloudy", "noisy", "clean"):
            if not np.array_equal(online_preview[key][index], ema_preview[key][index]):
                raise ValueError("online and EMA previews used different inputs")
        rows.append(
            np.concatenate(
                [
                    online_preview["cloudy"][index],
                    online_preview["noisy"][index],
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
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = load_json(root / "configs/teacher_training.json")
    validate_config(config)
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    protocol = load_json(root / "configs/data_protocol.json")
    subset_config = protocol["subsets"][args.subset]
    dataset = PairedRGBDataset(
        root / subset_config["test_manifest"],
        pair_transform=paired_model_transform,
        image_size=config["image_size"],
    )
    if len(dataset) != subset_config["test_count"]:
        raise AssertionError("test dataset count differs from fixed protocol")

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    validate_checkpoint(checkpoint, args.subset, config, args.expected_epoch)
    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]
    online_state = checkpoint.pop("model")
    ema_state = checkpoint.pop("ema_model")
    distance = state_distance(online_state, ema_state)
    del checkpoint
    gc.collect()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    model = build_model(root, config)
    model.load_state_dict(online_state)
    online_records, online_preview = evaluate_weights(model, dataset, config, "online")
    model.load_state_dict(ema_state)
    ema_records, ema_preview = evaluate_weights(model, dataset, config, "ema")
    del online_state, ema_state
    gc.collect()

    preview_path = args.output_dir / "online_vs_ema.png"
    save_preview(preview_path, comparison_rows(online_preview, ema_preview))
    result = {
        "status": "completed",
        "scope": "held_out_teacher_denoiser_evaluation_not_full_diffusion_sampling",
        "subset": args.subset,
        "checkpoint": str(checkpoint_path),
        "epoch": epoch,
        "global_step": global_step,
        "test_samples": len(dataset),
        "prediction_clamped": True,
        "preview_columns": ["cloudy", "noisy", "online", "ema", "clean"],
        "parameter_distance": distance,
        "metrics": {"online": online_records, "ema": ema_records},
        "deltas": metric_deltas(online_records, ema_records),
        "evaluation_seconds": time.monotonic() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    write_json(args.output_dir / "online_vs_ema.json", result)
    print(result)


if __name__ == "__main__":
    main()
