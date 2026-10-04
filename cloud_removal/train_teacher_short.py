"""Resumable short-run trainer for the conditional EDM + HDiT teacher."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset

from .data import PairedRGBDataset, paired_model_transform
from .edm import EDMInputScaling, EDMTeacher
from .hdit_adapter import ConditionedHDiT
from .hdit_factory import build_hdit
from .training import teacher_loss


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def edm_weights(sigmas, sigma_data):
    return (sigmas.square() + sigma_data**2) / (sigmas * sigma_data).square()


def sample_sigmas(batch_size, config, device):
    values = torch.randn(batch_size, device=device)
    values = (values * config["log_normal_std"] + config["log_normal_mean"]).exp()
    return values.clamp(config["minimum"], config["maximum"])


def append_jsonl(path, record):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def tensor_to_uint8(image):
    image = image.detach().float().clamp(-1, 1)
    return image.add(1).mul(127.5).round().byte().permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def save_fixed_samples(model, dataset, output_dir, step, seed):
    clean = torch.stack([dataset[index]["clean"] for index in range(2)]).cuda()
    cloudy = torch.stack([dataset[index]["cloudy"] for index in range(2)]).cuda()
    generator = torch.Generator(device="cpu").manual_seed(seed + 100_000)
    noise = torch.randn(clean.shape, generator=generator).cuda()
    sigma = torch.ones(clean.shape[0], device="cuda")
    noisy = clean + noise
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(noisy, sigma, cloudy)
    mse = (output.float() - clean).square().mean().item()
    rows = []
    for index in range(clean.shape[0]):
        columns = [cloudy[index], noisy[index], output[index], clean[index]]
        rows.append(np.concatenate([tensor_to_uint8(value) for value in columns], axis=1))
    grid = np.concatenate(rows, axis=0)
    path = output_dir / f"samples_step_{step:04d}.png"
    Image.fromarray(grid, mode="RGB").save(path)
    model.train()
    return {"step": step, "mse": mse, "file": path.name}


def move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_checkpoint(path, model, optimizer, step, config, subset, resumed_from):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": 1,
            "subset": subset,
            "step": step,
            "resumed_from": resumed_from,
            "config": config,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
        },
        temporary,
    )
    os.replace(temporary, path)


def validate_config(config):
    if config["status"] != "experimental_baseline_not_reported_by_thesis":
        raise ValueError("short-run config must remain explicitly experimental")
    if config["precision"] != "bfloat16" or not config["gradient_checkpointing"]:
        raise ValueError("this tested short-run entry requires BF16 and checkpointing")
    if config["batch_size"] != 4 or config["image_size"] != 512:
        raise ValueError("short-run input contract changed")
    sigma = config["sigma"]
    if not 0 < sigma["minimum"] < sigma["maximum"] or sigma["data"] <= 0:
        raise ValueError("invalid sigma configuration")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=("CUHK-CR1", "CUHK-CR2"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = load_json(root / "configs/teacher_short_run.json")
    validate_config(config)
    target_steps = args.max_steps or config["maximum_steps"]
    if target_steps < 1 or target_steps > config["maximum_steps"]:
        parser.error("--max-steps must be within the configured maximum")

    output_dir = args.output_dir.resolve()
    checkpoint_path = output_dir / "checkpoint.pt"
    metrics_path = output_dir / "metrics.jsonl"
    samples_path = output_dir / "sample_metrics.jsonl"
    summary_path = output_dir / "summary.json"
    if args.resume is None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError("fresh output directory is not empty")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_config.json").write_text(
            json.dumps({"subset": args.subset, **config}, indent=2), encoding="utf-8"
        )
        (output_dir / "sample_layout.json").write_text(
            json.dumps({"columns": ["cloudy", "noisy_sigma_1", "prediction", "clean"]}, indent=2),
            encoding="utf-8",
        )
    elif not args.resume.is_file():
        raise FileNotFoundError(args.resume)

    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    protocol = load_json(root / "configs/data_protocol.json")
    subset_config = protocol["subsets"][args.subset]
    dataset = PairedRGBDataset(
        root / subset_config["train_manifest"],
        pair_transform=paired_model_transform,
        image_size=config["image_size"],
    )
    if len(dataset) != subset_config["train_count"]:
        raise AssertionError("dataset count differs from fixed protocol")

    model_spec = load_json(root / config["model_config"])
    hdit = build_hdit(model_spec).cuda()
    model = EDMTeacher(
        ConditionedHDiT(hdit, input_transform=EDMInputScaling(config["sigma"]["data"])),
        sigma_data=config["sigma"]["data"],
    )
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        betas=tuple(optimizer_config["betas"]),
        weight_decay=optimizer_config["weight_decay"],
    )

    start_step = 0
    resumed_from = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if checkpoint["subset"] != args.subset or checkpoint["config"] != config:
            raise ValueError("checkpoint identity or configuration mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        move_optimizer_state(optimizer, "cuda")
        start_step = checkpoint["step"]
        resumed_from = start_step
        torch.set_rng_state(checkpoint["cpu_rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()
        existing = read_jsonl(metrics_path)
        if not existing or existing[-1]["step"] != start_step:
            raise ValueError("metrics and checkpoint steps do not agree")
    else:
        sample_record = save_fixed_samples(model, dataset, output_dir, 0, config["seed"])
        append_jsonl(samples_path, sample_record)

    if start_step >= target_steps:
        raise ValueError("checkpoint is already at or beyond the requested target")

    permutation_generator = torch.Generator().manual_seed(config["seed"])
    indices = torch.randperm(len(dataset), generator=permutation_generator).tolist()
    loader_generator = torch.Generator().manual_seed(config["seed"] + 1)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
        generator=loader_generator,
    )
    if target_steps > len(loader):
        raise ValueError("short run unexpectedly crosses an epoch boundary")

    iterator = iter(loader)
    for _ in range(start_step):
        next(iterator)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    sigma_config = config["sigma"]
    training_started = time.monotonic()

    import k_diffusion as K

    for step in range(start_step + 1, target_steps + 1):
        batch = next(iterator)
        clean = batch["clean"].to("cuda", non_blocking=True)
        cloudy = batch["cloudy"].to("cuda", non_blocking=True)
        sigmas = sample_sigmas(clean.shape[0], sigma_config, "cuda")
        noise = torch.randn_like(clean)
        weights = edm_weights(sigmas, sigma_config["data"])
        optimizer.zero_grad(set_to_none=True)
        step_started = time.monotonic()
        with K.models.checkpointing(True):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = teacher_loss(
                    model,
                    clean,
                    cloudy,
                    sigmas,
                    noise,
                    weights=weights,
                    reduction=config["loss"]["reduction"],
                )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at step {step}")
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError(f"missing or nonfinite gradient at step {step}")
        grad_norm = torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(gradient.detach().float()) for gradient in gradients])
        )
        optimizer.step()
        torch.cuda.synchronize()
        record = {
            "step": step,
            "loss": loss.detach().item(),
            "grad_norm": grad_norm.item(),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "sigma_min": sigmas.min().item(),
            "sigma_mean": sigmas.mean().item(),
            "sigma_max": sigmas.max().item(),
            "allocated_mib": torch.cuda.memory_allocated() / 2**20,
            "reserved_mib": torch.cuda.memory_reserved() / 2**20,
            "step_seconds": time.monotonic() - step_started,
        }
        append_jsonl(metrics_path, record)
        if step == 1 or step % 10 == 0 or step == target_steps:
            print(json.dumps(record), flush=True)

        if step in config["sample_steps"] or step == target_steps:
            sample_record = save_fixed_samples(model, dataset, output_dir, step, config["seed"])
            append_jsonl(samples_path, sample_record)
        if step % config["checkpoint_interval"] == 0 or step == target_steps:
            save_checkpoint(
                checkpoint_path, model, optimizer, step, config, args.subset, resumed_from
            )

    records = read_jsonl(metrics_path)
    sample_records = read_jsonl(samples_path)
    first = records[:20]
    last = records[-20:]
    first_mean = sum(item["loss"] for item in first) / len(first)
    last_mean = sum(item["loss"] for item in last) / len(last)
    summary = {
        "status": "completed",
        "subset": args.subset,
        "start_step": start_step,
        "end_step": target_steps,
        "resumed_from": resumed_from,
        "records": len(records),
        "first_20_loss_mean": first_mean,
        "last_20_loss_mean": last_mean,
        "last_below_first": last_mean < first_mean,
        "all_finite": all(
            math.isfinite(item["loss"]) and math.isfinite(item["grad_norm"])
            for item in records
        ),
        "fixed_sample_metrics": sample_records,
        "fixed_sample_mse_improved": sample_records[-1]["mse"] < sample_records[0]["mse"],
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "training_seconds_this_invocation": time.monotonic() - training_started,
        "checkpoint": str(checkpoint_path),
        "checkpoint_size_mib": checkpoint_path.stat().st_size / 2**20,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
