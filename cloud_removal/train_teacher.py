"""Epoch-based trainer for the experimental conditional EDM + HDiT teacher.

The thesis does not report these optimization choices. This entry point keeps
them explicit, trains CUHK-CR1 and CUHK-CR2 separately, and resumes only at
completed epoch checkpoints.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
from .teacher_lr_transition import validate_lr_transition, set_optimizer_lr
from .teacher_lr_reheat import validate_reheat_transition, parameter_snapshot, update_statistics
from .teacher_cosine import validate_schedule, validate_cosine_transition, learning_rate_at
from .teacher_reconstruction import reconstruction_weight, validate_reconstruction_transition
from .teacher_weighting import weighting_mode, denoising_weights, validate_soft_min_snr_transition
from .teacher_stratified import (
    sampling_mode, sample_training_sigmas, restore_sampler_rng, validate_stratified_transition,
)
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset

from .data import PairedRGBDataset, paired_model_transform
from .edm import EDMInputScaling, EDMTeacher
from .hdit_adapter import ConditionedHDiT
from .hdit_factory import build_hdit
from .image_metrics import mse_per_sample, psnr_per_sample, ssim_per_sample
from .teacher_checkpoint import (
    save_ema_milestone,
    save_rotating_checkpoint,
    steps_per_epoch,
)
from .train_teacher_short import (
    append_jsonl,
    edm_weights,
    load_json,
    move_optimizer_state,
    read_jsonl,
    sample_sigmas,
    tensor_to_uint8,
)
from .training import make_target, teacher_loss, update_ema
from .teacher_noise_ablation import (
    replace_with_high_noise, validate_mixture, validate_noise_transition,
)


def validate_config(config):
    sampling_mode(config)
    weighting_mode(config)
    reconstruction_weight(config)
    validate_schedule(config)
    if config.get("backbone_type", "hdit") not in ("hdit", "unet"):
        raise ValueError("unknown teacher backbone")
    if config["status"] != "experimental_baseline_not_reported_by_thesis":
        raise ValueError("undocumented training settings must remain explicitly experimental")
    if config["precision"] != "bfloat16" or not config["gradient_checkpointing"]:
        raise ValueError("the tested hardware contract requires BF16 and checkpointing")
    if config["batch_size"] != 4 or config["image_size"] != 512:
        raise ValueError("the tested batch and image-size contract changed")
    if config["drop_last"]:
        raise ValueError("formal training must retain every fixed-protocol sample")
    if not 0 <= config["ema_decay"] < 1:
        raise ValueError("EMA decay must be in [0, 1)")
    maximum = config["maximum_epochs"]
    stages = config["stage_epochs"]
    if maximum < 1 or stages != sorted(set(stages)) or not stages or stages[-1] != maximum:
        raise ValueError("stage epochs must be unique, increasing and end at maximum_epochs")
    for name in (
        "evaluation_interval_epochs",
        "checkpoint_interval_epochs",
        "log_interval_steps",
    ):
        if config[name] < 1:
            raise ValueError(f"{name} must be positive")
    monitor = config["monitor"]
    if monitor["sample_count"] < 1 or not 0 < monitor["preview_count"] <= monitor["sample_count"]:
        raise ValueError("invalid monitor sample counts")
    sigmas = monitor["sigmas"]
    if not sigmas or any(not math.isfinite(value) or value <= 0 for value in sigmas):
        raise ValueError("monitor sigmas must be finite and positive")
    if monitor["preview_sigma"] not in sigmas:
        raise ValueError("preview_sigma must be one of the monitor sigmas")
    sigma = config["sigma"]
    validate_mixture(sigma)
    if not 0 < sigma["minimum"] < sigma["maximum"] or sigma["data"] <= 0:
        raise ValueError("invalid sigma configuration")
    checkpoint = config["checkpoint"]
    if not checkpoint["keep_previous"] or checkpoint["milestone_weights"] != "ema_only":
        raise ValueError("formal checkpoint rotation contract changed")


def validate_epoch_extension(saved_config, current_config, allow_learning_rate_change=False):
    maximum = current_config["maximum_epochs"]
    if maximum <= saved_config["maximum_epochs"]:
        raise ValueError("epoch extension must increase maximum_epochs")
    expected = dict(saved_config)
    expected["maximum_epochs"] = maximum
    expected["stage_epochs"] = [*saved_config["stage_epochs"], maximum]
    if allow_learning_rate_change:
        lr = current_config["optimizer"]["learning_rate"]
        if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not 0 < lr < float('inf'):
            raise ValueError("learning rate must be positive and finite")
        expected["optimizer"] = {**saved_config["optimizer"], "learning_rate": lr}
    if current_config != expected:
        raise ValueError("epoch extension may only append the new final stage")


def validate_resume_config(saved_config, current_config, reset_ema_from_online):
    if saved_config == current_config:
        return
    comparable = dict(saved_config)
    comparable["ema_decay"] = current_config["ema_decay"]
    if comparable != current_config:
        raise ValueError("checkpoint configuration differs beyond EMA decay")
    if not reset_ema_from_online:
        raise ValueError("EMA decay transition requires --reset-ema-from-online")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def truncate_jsonl(path, predicate):
    path = Path(path)
    if not path.exists():
        return
    records = read_jsonl(path)
    retained = [record for record in records if predicate(record)]
    if len(retained) == len(records):
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in retained:
            handle.write(json.dumps(record) + "\n")
    os.replace(temporary, path)


def select_monitor_indices(dataset_size, count, seed):
    if count > dataset_size:
        raise ValueError("monitor sample count exceeds dataset size")
    generator = torch.Generator().manual_seed(seed + 300_000)
    return torch.randperm(dataset_size, generator=generator)[:count].tolist()


def build_model(root, config):
    model_spec = load_json(root / config["model_config"])
    transform = EDMInputScaling(config["sigma"]["data"])
    if config.get("backbone_type", "hdit") == "unet":
        from .unet_adapter import ConditionedUNet, build_unet
        backbone = ConditionedUNet(build_unet(model_spec).cuda(), input_transform=transform,
                                  gradient_checkpointing=config["gradient_checkpointing"])
    else:
        backbone = ConditionedHDiT(build_hdit(model_spec).cuda(), input_transform=transform)
    return EDMTeacher(
        backbone,
        sigma_data=config["sigma"]["data"],
    )


def build_optimizer(model, config):
    optimizer = config["optimizer"]
    validate_schedule(config)
    if optimizer["type"] != "AdamW":
        raise ValueError("only AdamW is implemented")
    return torch.optim.AdamW(
        model.parameters(),
        lr=optimizer["learning_rate"],
        betas=tuple(optimizer["betas"]),
        weight_decay=optimizer["weight_decay"],
    )


def save_preview(path, rows):
    grid = np.concatenate(rows, axis=0)
    path = Path(path)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    Image.fromarray(grid, mode="RGB").save(temporary, format="PNG")
    os.replace(temporary, path)


@torch.no_grad()
def evaluate_monitor(model, dataset, indices, config, output_dir, epoch, global_step):
    monitor = config["monitor"]
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    was_training = model.training
    model.eval()
    records = []
    preview_rows = None
    for sigma_index, sigma_value in enumerate(monitor["sigmas"]):
        totals = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0}
        seen = 0
        noise_generator = torch.Generator(device="cpu").manual_seed(
            config["seed"] + 400_000 + sigma_index
        )
        for batch in loader:
            clean = batch["clean"].to("cuda", non_blocking=True)
            cloudy = batch["cloudy"].to("cuda", non_blocking=True)
            noise = torch.randn(clean.shape, generator=noise_generator).to("cuda")
            sigmas = torch.full((clean.shape[0],), sigma_value, device="cuda")
            noisy = clean + sigmas[:, None, None, None] * noise
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(noisy, sigmas, cloudy)
            prediction = prediction.float().clamp(-1, 1)
            clean_float = clean.float()
            batch_metrics = {
                "mse": mse_per_sample(prediction, clean_float),
                "psnr": psnr_per_sample(prediction, clean_float),
                "ssim": ssim_per_sample(prediction, clean_float),
            }
            for name, values in batch_metrics.items():
                totals[name] += values.sum().item()
            seen += clean.shape[0]
            if sigma_value == monitor["preview_sigma"] and preview_rows is None:
                preview_rows = []
                for index in range(min(monitor["preview_count"], clean.shape[0])):
                    columns = [cloudy[index], noisy[index], prediction[index], clean[index]]
                    preview_rows.append(
                        np.concatenate([tensor_to_uint8(value) for value in columns], axis=1)
                    )
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "weights": "ema",
            "source": "fixed_training_monitor_not_validation",
            "prediction_clamped": True,
            "sigma": sigma_value,
            "samples": seen,
            **{name: value / seen for name, value in totals.items()},
        }
        records.append(record)
    if preview_rows is None:
        raise RuntimeError("preview sigma did not produce a preview")
    latest_preview = Path(output_dir) / "monitor_latest.png"
    save_preview(latest_preview, preview_rows)
    if epoch in config["stage_epochs"]:
        shutil.copy2(latest_preview, Path(output_dir) / f"monitor_epoch_{epoch:04d}.png")
    if was_training:
        model.train()
    return records


def checkpoint_payload(*, subset, epoch, global_step, config, model, ema_model,
                       optimizer, loader_generator, resumed_from):
    return {
        "format": 1,
        "kind": "full_training_state",
        "subset": subset,
        "epoch": epoch,
        "global_step": global_step,
        "resumed_from": resumed_from,
        "config": config,
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
        "loader_rng_state": loader_generator.get_state(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=("CUHK-CR1", "CUHK-CR2"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-epoch", type=int, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/teacher_training.json"))
    parser.add_argument("--fork-from", type=Path)
    parser.add_argument("--allow-epoch-extension", action="store_true")
    parser.add_argument("--allow-noise-distribution-change", action="store_true")
    parser.add_argument("--allow-learning-rate-change", action="store_true")
    parser.add_argument("--allow-cosine-transition", action="store_true")
    parser.add_argument("--allow-reconstruction-transition", action="store_true")
    parser.add_argument("--allow-soft-min-snr-transition", action="store_true")
    parser.add_argument("--allow-stratified-transition", action="store_true")
    parser.add_argument("--allow-lr-reheat-transition", action="store_true")
    parser.add_argument("--lr-update-diagnostics", action="store_true")
    parser.add_argument("--reset-ema-from-online", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = load_json(root / args.config)
    validate_config(config)
    if args.allow_lr_reheat_transition and (
        args.fork_from is None or args.reset_ema_from_online or args.allow_epoch_extension
        or args.allow_learning_rate_change or args.allow_noise_distribution_change
        or args.allow_cosine_transition or args.allow_reconstruction_transition
        or args.allow_soft_min_snr_transition or args.allow_stratified_transition
    ):
        parser.error("LR reheat requires a fresh fork and no other transitions")
    if args.lr_update_diagnostics and not args.allow_lr_reheat_transition:
        parser.error("update diagnostics are scoped to the LR pilot")
    if args.allow_stratified_transition and (
        args.fork_from is None or args.reset_ema_from_online or args.allow_epoch_extension
        or args.allow_learning_rate_change or args.allow_noise_distribution_change
        or args.allow_cosine_transition or args.allow_reconstruction_transition
        or args.allow_soft_min_snr_transition
    ):
        parser.error("stratified transition requires a fresh fork and no other transitions")
    if args.allow_soft_min_snr_transition and (
        args.fork_from is None or args.reset_ema_from_online or args.allow_epoch_extension
        or args.allow_learning_rate_change or args.allow_noise_distribution_change
        or args.allow_cosine_transition or args.allow_reconstruction_transition
    ):
        parser.error("soft_min_snr transition requires a fresh fork and no other transitions")
    if args.allow_reconstruction_transition and (
        args.fork_from is None or args.reset_ema_from_online or args.allow_epoch_extension
        or args.allow_learning_rate_change or args.allow_noise_distribution_change
        or args.allow_cosine_transition
    ):
        parser.error("reconstruction transition requires a fresh fork and no other transitions")
    if args.allow_cosine_transition and (
        args.fork_from is None or args.reset_ema_from_online or args.allow_epoch_extension
        or args.allow_learning_rate_change or args.allow_noise_distribution_change
    ):
        parser.error("cosine transition requires a fresh fork and no other transition flags")
    if args.allow_epoch_extension and (
        args.fork_from is None or args.reset_ema_from_online
        or args.allow_noise_distribution_change
    ):
        parser.error("epoch extension requires a fresh fork and no other transition")
    if args.allow_learning_rate_change and (
        args.fork_from is None or args.reset_ema_from_online
        or args.allow_noise_distribution_change
    ):
        parser.error("learning-rate transition requires a fresh fork and no other transition")
    if args.fork_from is not None and args.resume is not None:
        parser.error("--fork-from and --resume are mutually exclusive")
    if args.allow_noise_distribution_change and (
        args.fork_from is None or args.reset_ema_from_online
    ):
        parser.error("noise transition requires a fresh fork and cannot reset EMA")
    # This environment has no C compiler, so upstream torch.compile decorators
    # would only warn and fall back to the same eager execution used here.
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    if not 1 <= args.target_epoch <= config["maximum_epochs"]:
        parser.error("--target-epoch must be between 1 and maximum_epochs")
    if args.reset_ema_from_online and args.resume is None:
        parser.error("--reset-ema-from-online requires --resume")

    output_dir = args.output_dir.resolve()
    training_metrics_path = output_dir / "training_metrics.jsonl"
    epoch_metrics_path = output_dir / "epoch_metrics.jsonl"
    evaluation_metrics_path = output_dir / "evaluation_metrics.jsonl"
    summary_path = output_dir / "summary.json"
    if args.resume is None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError("fresh output directory is not empty")
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "run_config.json", {"subset": args.subset, **config})
        write_json(
            output_dir / "monitor_layout.json",
            {
                "columns": ["cloudy", "noisy", "ema_prediction", "clean"],
                "source": "fixed_training_monitor_not_validation",
            },
        )
    elif not args.resume.is_file():
        raise FileNotFoundError(args.resume)
    elif args.resume.resolve().parent != output_dir:
        raise ValueError("resume checkpoint must belong to the selected output directory")
    if args.fork_from is not None:
        if not args.fork_from.is_file():
            raise FileNotFoundError(args.fork_from)
        args.resume = args.fork_from

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
    expected_steps = steps_per_epoch(
        len(dataset), config["batch_size"], drop_last=config["drop_last"]
    )
    monitor_indices = select_monitor_indices(
        len(dataset), config["monitor"]["sample_count"], config["seed"]
    )
    monitor_manifest = {
        "indices": monitor_indices,
        "ids": [dataset.samples[index]["id"] for index in monitor_indices],
    }
    monitor_manifest_path = output_dir / "monitor_samples.json"
    if monitor_manifest_path.exists():
        if load_json(monitor_manifest_path) != monitor_manifest:
            raise ValueError("fixed monitor sample identity changed")
    else:
        write_json(monitor_manifest_path, monitor_manifest)

    model = build_model(root, config)
    ema_model = make_target(model)
    optimizer = build_optimizer(model, config)
    loader_generator = torch.Generator().manual_seed(config["seed"] + 1)
    mixture_generator = torch.Generator().manual_seed(config["seed"] + 900_000)
    stratified_generator = torch.Generator().manual_seed(config["seed"] + 910_000)
    start_epoch = 0
    global_step = 0
    resumed_from = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if (
            checkpoint["kind"] != "full_training_state"
            or checkpoint["subset"] != args.subset
        ):
            raise ValueError("checkpoint identity mismatch")
        if args.allow_lr_reheat_transition:
            validate_reheat_transition(checkpoint["config"], config, checkpoint["global_step"])
        elif args.allow_stratified_transition:
            validate_stratified_transition(checkpoint["config"], config)
        elif args.allow_soft_min_snr_transition:
            validate_soft_min_snr_transition(checkpoint["config"], config)
        elif args.allow_reconstruction_transition:
            validate_reconstruction_transition(checkpoint["config"], config)
        elif args.allow_cosine_transition:
            validate_cosine_transition(checkpoint["config"], config, checkpoint["global_step"])
            schedule = config["learning_rate_schedule"]
            if schedule['end_step'] != config['maximum_epochs'] * expected_steps:
                raise ValueError('cosine horizon must match maximum_epochs')
        elif args.allow_epoch_extension:
            validate_epoch_extension(checkpoint["config"], config, args.allow_learning_rate_change)
        elif args.allow_learning_rate_change:
            validate_lr_transition(checkpoint["config"], config)
        elif args.allow_noise_distribution_change:
            validate_noise_transition(checkpoint["config"], config)
        else:
            validate_resume_config(
                checkpoint["config"], config, args.reset_ema_from_online
            )
        previous_ema_decay = checkpoint["config"]["ema_decay"]
        model.load_state_dict(checkpoint["model"])
        if args.reset_ema_from_online:
            ema_model.load_state_dict(checkpoint["model"])
        else:
            ema_model.load_state_dict(checkpoint["ema_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if args.allow_learning_rate_change or args.allow_lr_reheat_transition:
            set_optimizer_lr(optimizer, config["optimizer"]["learning_rate"])
        move_optimizer_state(optimizer, "cuda")
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        resumed_from = {
            "lr_reheat_transition": args.allow_lr_reheat_transition,
            "lr_update_diagnostics": args.lr_update_diagnostics,
            "stratified_transition": args.allow_stratified_transition,
            "soft_min_snr_transition": args.allow_soft_min_snr_transition,
            "reconstruction_transition": args.allow_reconstruction_transition,
            "cosine_transition": args.allow_cosine_transition,
            "epoch_extension": args.allow_epoch_extension,
            "checkpoint": str(args.resume.resolve()),
            "noise_distribution_transition": args.allow_noise_distribution_change,
            "learning_rate_transition": args.allow_learning_rate_change,
            "epoch": start_epoch,
            "global_step": global_step,
            "ema_reset_from_online": args.reset_ema_from_online,
            "previous_ema_decay": previous_ema_decay,
            "ema_decay": config["ema_decay"],
        }
        torch.set_rng_state(checkpoint["cpu_rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        loader_generator.set_state(checkpoint["loader_rng_state"])
        restore_sampler_rng(stratified_generator, checkpoint)
        if "mixture_rng_state" in checkpoint:
            mixture_generator.set_state(checkpoint["mixture_rng_state"])
        elif "high_noise_mixture" in checkpoint["config"]["sigma"]:
            raise ValueError("mixture checkpoint lacks its RNG state")
        if args.reset_ema_from_online:
            write_json(
                output_dir / f"phase_from_epoch_{start_epoch:04d}.json",
                {
                    "status": "experimental_ema_transition_not_reported_by_thesis",
                    "subset": args.subset,
                    "start_epoch": start_epoch,
                    "target_epoch": args.target_epoch,
                    "resume_checkpoint": str(args.resume.resolve()),
                    "ema_reset_from_online": True,
                    "previous_ema_decay": previous_ema_decay,
                    "ema_decay": config["ema_decay"],
                    "config": config,
                },
            )
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()
        truncate_jsonl(training_metrics_path, lambda record: record["step"] <= global_step)
        truncate_jsonl(epoch_metrics_path, lambda record: record["epoch"] <= start_epoch)
        truncate_jsonl(evaluation_metrics_path, lambda record: record["epoch"] <= start_epoch)
    if start_epoch >= args.target_epoch:
        raise ValueError("checkpoint is already at or beyond the requested target epoch")

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=config["drop_last"],
        generator=loader_generator,
        persistent_workers=config["num_workers"] > 0,
    )
    if len(loader) != expected_steps:
        raise AssertionError("data loader step count differs from the fixed protocol")

    if start_epoch == 0:
        initial_records = evaluate_monitor(
            ema_model, dataset, monitor_indices, config, output_dir, 0, 0
        )
        for record in initial_records:
            append_jsonl(evaluation_metrics_path, record)

    import k_diffusion as K

    if args.resume is None and "training_rng_seed" in config:
        torch.manual_seed(config["training_rng_seed"])
        torch.cuda.manual_seed_all(config["training_rng_seed"])
    torch.cuda.reset_peak_memory_stats()
    model.train()
    sigma_config = config["sigma"]
    invocation_started = time.monotonic()
    for epoch in range(start_epoch + 1, args.target_epoch + 1):
        epoch_started = time.monotonic()
        epoch_loss = 0.0
        epoch_samples = 0
        for batch_index, batch in enumerate(loader, start=1):
            clean = batch["clean"].to("cuda", non_blocking=True)
            cloudy = batch["cloudy"].to("cuda", non_blocking=True)
            sigmas, sigma_strata = sample_training_sigmas(
                clean.shape[0], config, "cuda", stratified_generator)
            sigmas = replace_with_high_noise(sigmas, sigma_config, mixture_generator)
            noise_rng_hash = hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()
            noise = torch.randn_like(clean)
            weights = denoising_weights(sigmas, sigma_config["data"], weighting_mode(config))
            optimizer.zero_grad(set_to_none=True)
            step_started = time.monotonic()
            with K.models.checkpointing(True):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, loss_components = teacher_loss(
                        model,
                        clean,
                        cloudy,
                        sigmas,
                        noise,
                        weights=weights,
                        reduction=config["loss"]["reduction"],
                        reconstruction_l1_weight=reconstruction_weight(config),
                        return_components=True,
                        return_per_sample=args.lr_update_diagnostics,
                    )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at step {global_step + 1}")
            loss.backward()
            gradients = [
                parameter.grad for parameter in model.parameters() if parameter.grad is not None
            ]
            if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
                raise FloatingPointError(f"missing or nonfinite gradient at step {global_step + 1}")
            grad_norm = torch.linalg.vector_norm(
                torch.stack(
                    [torch.linalg.vector_norm(gradient.detach().float()) for gradient in gradients]
                )
            )
            if config['learning_rate_schedule'] != 'constant':
                set_optimizer_lr(optimizer, learning_rate_at(config, global_step))
            before_update = parameter_snapshot(model) if (
                args.lr_update_diagnostics and batch_index in (1, expected_steps)) else None
            optimizer.step()
            update_record = update_statistics(model, before_update) if before_update is not None else None
            del before_update
            update_ema(ema_model, model, decay=config["ema_decay"])
            global_step += 1
            torch.cuda.synchronize()
            batch_size = clean.shape[0]
            epoch_loss += loss.detach().item() * batch_size
            epoch_samples += batch_size
            record = {
                "sample_ids": list(batch["id"]),
                "image_noise_rng_sha256": noise_rng_hash,
                "sigma_ge_3_count": int((sigmas >= 3).sum().item()),
                "epoch": epoch,
                "batch": batch_index,
                "step": global_step,
                "batch_size": batch_size,
                "loss": loss.detach().item(),
                "edm_loss": loss_components['edm_loss'].item() if weighting_mode(config) == 'edm' else None,
                "weighted_denoising_loss": loss_components['edm_loss'].item(),
                "loss_weighting": weighting_mode(config),
                "sigma_sampling": sampling_mode(config),
                "sigma_strata": sigma_strata,
                "sigma_values": sigmas.detach().cpu().tolist(),
                "loss_weight_min": weights.min().item(),
                "loss_weight_mean": weights.mean().item(),
                "loss_weight_max": weights.max().item(),
                "reconstruction_l1": loss_components['reconstruction_l1'].item(),
                "reconstruction_l1_weight": reconstruction_weight(config),
                "grad_norm": grad_norm.item(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "sigma_min": sigmas.min().item(),
                "sigma_mean": sigmas.mean().item(),
                "sigma_max": sigmas.max().item(),
                "allocated_mib": torch.cuda.memory_allocated() / 2**20,
                "reserved_mib": torch.cuda.memory_reserved() / 2**20,
                "step_seconds": time.monotonic() - step_started,
            }
            append_jsonl(training_metrics_path, record)
            if args.lr_update_diagnostics:
                append_jsonl(output_dir / 'lr_diagnostics.jsonl', {
                    'epoch': epoch, 'step': global_step, 'sample_ids': list(batch['id']),
                    'sigma_values': record['sigma_values'],
                    'denoising_mse_per_sample': loss_components['denoising_mse_per_sample'].cpu().tolist(),
                    'learning_rate': record['learning_rate'], 'parameter_update': update_record,
                })
            if global_step == 1 or global_step % config["log_interval_steps"] == 0:
                print(json.dumps(record), flush=True)
        if epoch_samples != len(dataset):
            raise AssertionError("an epoch did not retain every training sample")
        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            "samples": epoch_samples,
            "mean_loss": epoch_loss / epoch_samples,
            "epoch_seconds": time.monotonic() - epoch_started,
        }
        append_jsonl(epoch_metrics_path, epoch_record)
        print(json.dumps({"epoch_summary": epoch_record}), flush=True)

        should_evaluate = (
            epoch % config["evaluation_interval_epochs"] == 0
            or epoch == args.target_epoch
        )
        if should_evaluate:
            evaluation_records = evaluate_monitor(
                ema_model, dataset, monitor_indices, config, output_dir, epoch, global_step
            )
            for record in evaluation_records:
                append_jsonl(evaluation_metrics_path, record)
            print(json.dumps({"monitor": evaluation_records}), flush=True)

        should_checkpoint = (
            epoch % config["checkpoint_interval_epochs"] == 0
            or epoch == args.target_epoch
        )
        if should_checkpoint:
            payload = checkpoint_payload(
                subset=args.subset,
                epoch=epoch,
                global_step=global_step,
                config=config,
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                loader_generator=loader_generator,
                resumed_from=resumed_from,
            )
            payload["mixture_rng_state"] = mixture_generator.get_state()
            payload["stratified_rng_state"] = stratified_generator.get_state()
            save_rotating_checkpoint(
                output_dir,
                payload,
                keep_previous=config["checkpoint"]["keep_previous"],
            )
            del payload
        if epoch in config["stage_epochs"]:
            save_ema_milestone(
                output_dir,
                subset=args.subset,
                epoch=epoch,
                global_step=global_step,
                config=config,
                ema_model=ema_model,
            )

    latest_path = output_dir / "latest.pt"
    summary = {
        "status": "completed_to_requested_epoch",
        "subset": args.subset,
        "start_epoch": start_epoch,
        "end_epoch": args.target_epoch,
        "global_step": global_step,
        "steps_per_epoch": expected_steps,
        "samples_per_epoch": len(dataset),
        "drop_last": config["drop_last"],
        "ema_decay": config["ema_decay"],
        "resumed_from": resumed_from,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "training_seconds_this_invocation": time.monotonic() - invocation_started,
        "latest_checkpoint": str(latest_path),
        "latest_checkpoint_size_mib": latest_path.stat().st_size / 2**20,
    }
    write_json(summary_path, summary)
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
