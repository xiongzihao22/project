"""Resumable experimental consistency-distillation student trainer.

The thesis specifies the student optimizer, learning rate, batch size and 600
epochs, but not the teacher solver, grid, initialization or EMA decay. All
missing choices are required in an explicitly experimental config file.
"""
import argparse
import gc
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from .data import PairedRGBDataset, paired_model_transform
from .edm import EDMInputScaling, identity_inputs
from .hdit_adapter import ConditionedHDiT
from .hdit_factory import build_hdit
from .student_trajectory import (
    karras_noise_levels,
    sample_adjacent_sigmas,
    teacher_pf_ode_step,
)
from .teacher_checkpoint import save_rotating_checkpoint, steps_per_epoch
from .train_teacher import build_model as build_teacher
from .train_teacher import truncate_jsonl, write_json
from .train_teacher_short import append_jsonl, load_json, move_optimizer_state
from .training import ConsistencyFunction, distillation_loss, make_target, update_ema
from .student_ema import validate_ema_config, normalize_ema_transition, inference_ema_state
from .student_endpoint import validate_endpoint_config, normalize_endpoint_transition, endpoint_l1


def validate_config(config):
    validate_endpoint_config(config)
    validate_ema_config(config)
    backbone = config.get("student_backbone", "hdit")
    if backbone not in ("hdit", "unet"):
        raise ValueError("unknown student backbone")
    if backbone == "unet" and (
            config.get("student_initialization") not in ("random", "teacher_backbone")
            or not config.get("teacher_model_config")):
        raise ValueError("U-Net requires an approved initialization and explicit teacher spec")
    weight = config.get("reconstruction_weight", 0.0)
    if (isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not math.isfinite(weight) or weight < 0):
        raise ValueError("invalid reconstruction weight")
    if config.get("status") != "experimental_student_distillation_not_reported_by_thesis":
        raise ValueError("student training choices must be explicitly experimental")
    if config["precision"] != "bfloat16" or not config["gradient_checkpointing"]:
        raise ValueError("the GPU contract requires BF16 and checkpointing")
    if config["image_size"] != 512 or config["batch_size"] != 4 or config["drop_last"]:
        raise ValueError("fixed image, batch or complete-data contract changed")
    if config["num_workers"] != 4 or config["augmentation"] != "none":
        raise ValueError("fixed loader and augmentation contract changed")
    if config["teacher_weights"] not in ("online", "ema"):
        raise ValueError("choose teacher weights explicitly")
    if config["student_initialization"] not in ("teacher_backbone", "random"):
        raise ValueError("choose student initialization explicitly")
    if config["student_input_scaling"] not in ("edm", "identity"):
        raise ValueError("choose student input scaling explicitly")
    if config["trajectory_solver"] not in ("euler", "heun"):
        raise ValueError("choose teacher trajectory solver explicitly")
    sigma = config["sigma"]
    if config.get("high_noise_probability", 0.0) not in (0.0, 0.5):
        raise ValueError("unapproved high-noise probability")
    if config.get("high_noise_probability", 0.0) and sigma["grid_levels"] != 18:
        raise ValueError("high-noise mixture requires 18 grid levels")
    if not 0 < sigma["minimum"] < sigma["maximum"] or sigma["data"] <= 0:
        raise ValueError("invalid noise bounds")
    if not math.isfinite(sigma["rho"]) or sigma["rho"] <= 0:
        raise ValueError("invalid rho")
    if not isinstance(sigma["grid_levels"], int) or sigma["grid_levels"] < 2:
        raise ValueError("noise grid needs at least two levels")
    if config["loss_reduction"] not in ("pixel_mean", "squared_l2"):
        raise ValueError("choose L2 reduction explicitly")
    if not 0 <= config["ema_decay"] < 1:
        raise ValueError("invalid EMA decay")
    optimizer = config["optimizer"]
    if optimizer["type"] != "AdamW" or optimizer["learning_rate"] not in (1e-5, 3e-6):
        raise ValueError("student AdamW LR must be an approved experimental value")
    if len(optimizer["betas"]) != 2 or not all(0 <= v < 1 for v in optimizer["betas"]):
        raise ValueError("invalid AdamW betas")
    if optimizer["weight_decay"] < 0 or not math.isfinite(optimizer["weight_decay"]):
        raise ValueError("invalid AdamW weight decay")
    if config["learning_rate_schedule"] != "constant":
        raise ValueError("only the explicit constant-LR pilot is implemented")
    gradient_clip_norm = config.get("gradient_clip_norm")
    if (gradient_clip_norm is not None
            and (isinstance(gradient_clip_norm, bool)
                 or not isinstance(gradient_clip_norm, (int, float))
                 or not math.isfinite(gradient_clip_norm)
                 or gradient_clip_norm <= 0)):
        raise ValueError("gradient clip norm must be null or a positive finite number")
    if not isinstance(config["maximum_epochs"], int) or config["maximum_epochs"] < 1:
        raise ValueError("invalid epoch limit")
    if not isinstance(config["log_interval_steps"], int) or config["log_interval_steps"] < 1:
        raise ValueError("invalid log interval")


def validate_resume_config(checkpoint_config, requested_config, *,
                           allow_gradient_clip_change=False, allow_reconstruction_change=False,
                           allow_learning_rate_change=False, allow_solver_change=False,
                           allow_noise_sampling_change=False, allow_teacher_change=False,
                           allow_ema_split=False, allow_endpoint_change=False):
    requested_config = normalize_endpoint_transition(checkpoint_config, requested_config, allow_endpoint_change)
    requested_config = normalize_ema_transition(checkpoint_config, requested_config, allow_ema_split)
    from .student_teacher_transition import normalized_resume_config
    requested_config = normalized_resume_config(checkpoint_config, requested_config, allow_teacher_change)
    validate_config(checkpoint_config)
    validate_config(requested_config)
    saved = {key: value for key, value in checkpoint_config.items()
             if key not in ("maximum_epochs", "gradient_clip_norm", "reconstruction_weight")}
    requested = {key: value for key, value in requested_config.items()
                 if key not in ("maximum_epochs", "gradient_clip_norm", "reconstruction_weight")}
    before_sampling = saved.pop("high_noise_probability", 0.0)
    after_sampling = requested.pop("high_noise_probability", 0.0)
    if before_sampling != after_sampling and not (
            allow_noise_sampling_change and before_sampling == 0.0 and after_sampling == 0.5):
        raise ValueError("noise sampling change is not authorized")
    before_lr = saved["optimizer"]["learning_rate"]
    after_lr = requested["optimizer"]["learning_rate"]
    if before_lr != after_lr and not (allow_learning_rate_change and before_lr == 1e-5 and after_lr == 3e-6):
        raise ValueError("learning rate change is not authorized")
    saved["optimizer"] = {**saved["optimizer"], "learning_rate": after_lr}
    if saved["trajectory_solver"] != requested["trajectory_solver"]:
        if not (allow_solver_change and saved["trajectory_solver"] == "euler"
                and requested["trajectory_solver"] == "heun"):
            raise ValueError("solver change is not authorized")
        saved["trajectory_solver"] = requested["trajectory_solver"]
    before = checkpoint_config.get("reconstruction_weight", 0.0)
    after = requested_config.get("reconstruction_weight", 0.0)
    if before != after and not (allow_reconstruction_change and before == 0 and after == 0.0001):
        raise ValueError("reconstruction loss change is not authorized")
    if saved != requested or requested_config["maximum_epochs"] < checkpoint_config["maximum_epochs"]:
        raise ValueError("resume may only increase maximum_epochs")
    saved_clip = checkpoint_config.get("gradient_clip_norm")
    requested_clip = requested_config.get("gradient_clip_norm")
    if saved_clip != requested_clip:
        if not (allow_gradient_clip_change
                and saved_clip is None and requested_clip is not None):
            raise ValueError("resume gradient clipping change is not authorized")


def validate_teacher_checkpoint(checkpoint, subset, config):
    if checkpoint.get("kind") != "full_training_state" or checkpoint.get("subset") != subset:
        raise ValueError("teacher checkpoint identity mismatch")
    if checkpoint.get("epoch") != config["teacher_epoch"]:
        raise ValueError("teacher checkpoint epoch mismatch")
    teacher_config = checkpoint["config"]
    if teacher_config["model_config"] != config.get("teacher_model_config", config["model_config"]):
        raise ValueError("teacher and student backbone specs differ")
    for key in ("minimum", "maximum", "data"):
        if teacher_config["sigma"][key] != config["sigma"][key]:
            raise ValueError("teacher and student noise constants differ")
    key = "model" if config["teacher_weights"] == "online" else "ema_model"
    if key not in checkpoint:
        raise ValueError("selected teacher weights are missing")
    return key


def restore_optimizer(optimizer, state, learning_rate):
    optimizer.load_state_dict(state)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def build_student(root, config):
    specification = load_json(root / config["model_config"])
    transform = (
        EDMInputScaling(config["sigma"]["data"])
        if config["student_input_scaling"] == "edm" else identity_inputs
    )
    if config.get("student_backbone", "hdit") == "unet":
        from .unet_adapter import ConditionedUNet, build_unet
        backbone = ConditionedUNet(build_unet(specification).cuda(), input_transform=transform,
                                  gradient_checkpointing=config["gradient_checkpointing"])
    else:
        backbone = ConditionedHDiT(build_hdit(specification).cuda(), input_transform=transform)
    return ConsistencyFunction(
        backbone,
        sigma_min=config["sigma"]["minimum"],
        sigma_data=config["sigma"]["data"],
    )


def initialize_student(student, teacher, mode):
    if mode == "teacher_backbone":
        if type(student.backbone) is not type(teacher.backbone):
            raise ValueError("teacher initialization requires matching backbone types")
        student.backbone.load_state_dict(teacher.backbone.state_dict(), strict=True)
    elif mode != "random":
        raise ValueError("invalid student initialization")


def initialize_from_teacher_state(student, checkpoint, subset, config):
    if config["student_initialization"] != "teacher_backbone":
        raise ValueError("explicit initialization requires teacher_backbone mode")
    init_config = {**config, "teacher_epoch": checkpoint.get("epoch")}
    key = validate_teacher_checkpoint(checkpoint, subset, init_config)
    prefix = "backbone."
    backbone = {name[len(prefix):]: value for name, value in checkpoint[key].items()
                if name.startswith(prefix)}
    student.backbone.load_state_dict(backbone, strict=True)


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def checkpoint_payload(*, subset, epoch, global_step, config, teacher_checkpoint,
                       student, target, optimizer, loader_generator, noise_generator):
    return {
        "format": 1,
        "kind": "student_distillation_full_training_state",
        "subset": subset,
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
        "teacher_checkpoint": str(teacher_checkpoint),
        "student": student.state_dict(),
        "target": target.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
        "loader_rng_state": loader_generator.get_state(),
        "noise_rng_state": noise_generator.get_state(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=("CUHK-CR1", "CUHK-CR2"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-epoch", type=int, required=True)
    parser.add_argument("--milestone-epochs", type=int, nargs="*", default=[])
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialization-teacher-checkpoint", type=Path)
    parser.add_argument("--allow-gradient-clip-change", action="store_true")
    parser.add_argument("--allow-reconstruction-change", action="store_true")
    parser.add_argument("--allow-learning-rate-change", action="store_true")
    parser.add_argument("--allow-solver-change", action="store_true")
    parser.add_argument("--allow-noise-sampling-change", action="store_true")
    parser.add_argument("--allow-teacher-change", action="store_true")
    parser.add_argument("--allow-ema-split", action="store_true")
    parser.add_argument("--allow-endpoint-change", action="store_true")
    parser.add_argument("--latest-only", action="store_true",
                        help="Keep only the latest full checkpoint in this run directory")
    args = parser.parse_args()
    if args.allow_endpoint_change and (args.resume is None or any((
            args.allow_ema_split, args.allow_teacher_change, args.allow_gradient_clip_change,
            args.allow_reconstruction_change, args.allow_learning_rate_change,
            args.allow_solver_change, args.allow_noise_sampling_change))):
        parser.error("endpoint supervision requires resume and no other transition")
    if args.allow_ema_split and (args.resume is None or any((
            args.allow_teacher_change, args.allow_gradient_clip_change,
            args.allow_reconstruction_change, args.allow_learning_rate_change,
            args.allow_solver_change, args.allow_noise_sampling_change))):
        parser.error("EMA split requires resume and no other training transition")
    if args.allow_teacher_change and (args.resume is None or any((
            args.allow_gradient_clip_change, args.allow_reconstruction_change,
            args.allow_learning_rate_change, args.allow_solver_change,
            args.allow_noise_sampling_change))):
        parser.error("teacher change requires resume and cannot combine other transitions")
    if args.allow_noise_sampling_change and args.resume is None:
        parser.error("--allow-noise-sampling-change requires --resume")
    if args.allow_solver_change and args.resume is None:
        parser.error("--allow-solver-change requires --resume")
    if args.allow_learning_rate_change and args.resume is None:
        parser.error("--allow-learning-rate-change requires --resume")
    if any(epoch < 1 or epoch > args.target_epoch for epoch in args.milestone_epochs):
        parser.error("milestone epochs must be within the requested training range")
    if args.allow_reconstruction_change and args.resume is None:
        parser.error("--allow-reconstruction-change requires --resume")
    if args.initialization_teacher_checkpoint is not None and args.resume is not None:
        parser.error("explicit initialization is only allowed for fresh training")
    if args.allow_gradient_clip_change and args.resume is None:
        parser.error("--allow-gradient-clip-change requires --resume")

    root = Path(__file__).resolve().parents[1]
    config = load_json(args.config)
    validate_config(config)
    if not 1 <= args.target_epoch <= config["maximum_epochs"]:
        parser.error("target epoch is outside configured maximum")
    if not torch.cuda.is_available():
        raise RuntimeError("student training requires CUDA")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    teacher_path = args.teacher_checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    metrics_path = output_dir / "training_metrics.jsonl"
    epochs_path = output_dir / "epoch_metrics.jsonl"

    if args.resume is None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError("fresh output directory is not empty")
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "run_config.json", {"subset": args.subset, **config})
    elif not args.resume.is_file() or args.resume.resolve().parent != output_dir:
        raise ValueError("resume checkpoint must exist inside selected output directory")

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
        raise AssertionError("fixed training split size changed")
    expected_steps = steps_per_epoch(len(dataset), config["batch_size"], drop_last=False)

    teacher_state = torch.load(teacher_path, map_location="cpu")
    teacher_key = validate_teacher_checkpoint(teacher_state, args.subset, config)
    teacher = build_teacher(root, teacher_state["config"])
    teacher.load_state_dict(teacher_state[teacher_key])
    teacher.eval().requires_grad_(False)
    del teacher_state
    gc.collect()
    student = build_student(root, config)
    initialize_student(student, teacher, config["student_initialization"])
    if args.initialization_teacher_checkpoint is not None:
        initial_state = torch.load(args.initialization_teacher_checkpoint, map_location="cpu")
        initialize_from_teacher_state(student, initial_state, args.subset, config)
        del initial_state
        gc.collect()
    if args.resume is None:
        write_json(output_dir / "initialization.json", {
            "teacher_checkpoint": str((args.initialization_teacher_checkpoint or teacher_path).resolve()),
            "student_state_sha256": state_digest(student.state_dict()),
        })
    target = make_target(student)
    inference_ema = make_target(student) if 'inference_ema_decay' in config else None
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=optimizer_config["learning_rate"],
        betas=tuple(optimizer_config["betas"]),
        weight_decay=optimizer_config["weight_decay"],
    )
    loader_generator = torch.Generator().manual_seed(config["seed"] + 1)
    noise_generator = torch.Generator().manual_seed(config["seed"] + 2)
    start_epoch = 0
    global_step = 0
    # Opt-in paired experiments must not inherit architecture-dependent RNG consumption.
    if args.resume is None and "training_rng_seed" in config:
        torch.manual_seed(config["training_rng_seed"])
        torch.cuda.manual_seed_all(config["training_rng_seed"])
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if (checkpoint.get("kind") != "student_distillation_full_training_state"
                or checkpoint.get("subset") != args.subset
                or (checkpoint.get("teacher_checkpoint") != str(teacher_path)
                    and not args.allow_teacher_change)):
            raise ValueError("student checkpoint identity mismatch")
        validate_resume_config(
            checkpoint.get("config"), config,
            allow_gradient_clip_change=args.allow_gradient_clip_change,
            allow_reconstruction_change=args.allow_reconstruction_change,
            allow_learning_rate_change=args.allow_learning_rate_change,
            allow_solver_change=args.allow_solver_change,
            allow_noise_sampling_change=args.allow_noise_sampling_change,
            allow_teacher_change=args.allow_teacher_change,
            allow_ema_split=args.allow_ema_split,
            allow_endpoint_change=args.allow_endpoint_change,
        )
        write_json(output_dir / "run_config.json", {
            "subset": args.subset,
            **config,
            "resume_checkpoint": str(args.resume.resolve()),
            "gradient_clip_change_authorized": args.allow_gradient_clip_change,
            "reconstruction_change_authorized": args.allow_reconstruction_change,
            "learning_rate_change_authorized": args.allow_learning_rate_change,
            "solver_change_authorized": args.allow_solver_change,
            "noise_sampling_change_authorized": args.allow_noise_sampling_change,
            "teacher_change_authorized": args.allow_teacher_change,
            "ema_split_authorized": args.allow_ema_split,
            "endpoint_change_authorized": args.allow_endpoint_change,
            "previous_teacher_checkpoint": checkpoint.get("teacher_checkpoint"),
        })
        student.load_state_dict(checkpoint["student"])
        target.load_state_dict(checkpoint["target"])
        if inference_ema is not None:
            inference_ema.load_state_dict(inference_ema_state(checkpoint))
        restore_optimizer(optimizer, checkpoint["optimizer"], config["optimizer"]["learning_rate"])
        move_optimizer_state(optimizer, "cuda")
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        torch.set_rng_state(checkpoint["cpu_rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        loader_generator.set_state(checkpoint["loader_rng_state"])
        noise_generator.set_state(checkpoint["noise_rng_state"])
        del checkpoint
        gc.collect()
        truncate_jsonl(metrics_path, lambda record: record["step"] <= global_step)
        truncate_jsonl(epochs_path, lambda record: record["epoch"] <= start_epoch)
    if start_epoch >= args.target_epoch:
        raise ValueError("checkpoint is already at or beyond requested epoch")

    loader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"], pin_memory=True, drop_last=False,
        generator=loader_generator, persistent_workers=True,
    )
    if len(loader) != expected_steps:
        raise AssertionError("dataloader omitted a fixed-protocol sample")
    levels = karras_noise_levels(
        config["sigma"]["grid_levels"], config["sigma"]["minimum"],
        config["sigma"]["maximum"], config["sigma"]["rho"], device="cuda",
    )
    import k_diffusion as K

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for epoch in range(start_epoch + 1, args.target_epoch + 1):
        student.train()
        target.eval()
        epoch_started = time.monotonic()
        epoch_loss = 0.0
        epoch_samples = 0
        for batch_index, batch in enumerate(loader, start=1):
            clean = batch["clean"].to("cuda", non_blocking=True)
            cloudy = batch["cloudy"].to("cuda", non_blocking=True)
            high, low = sample_adjacent_sigmas(
                levels, clean.shape[0], generator=noise_generator,
                high_noise_probability=config.get("high_noise_probability", 0.0),
            )
            noise = torch.randn_like(clean)
            input_audit = {
                "sample_ids": list(batch["id"]),
                "image_noise_rng_sha256": hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest(),
                "sigma_high": high.detach().cpu().tolist(),
                "sigma_low": low.detach().cpu().tolist(),
            }
            optimizer.zero_grad(set_to_none=True)
            step_started = time.monotonic()

            def trajectory(model, state, sigma_high, sigma_low, condition):
                return teacher_pf_ode_step(
                    model, state, sigma_high, sigma_low, condition,
                    solver=config["trajectory_solver"],
                )

            with K.models.checkpointing(True):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, loss_components = distillation_loss(
                        student, target, teacher, clean, cloudy, high, low, noise,
                        trajectory_step=trajectory,
                        reduction=config["loss_reduction"],
                        reconstruction_weight=config.get("reconstruction_weight", 0.0),
                        return_components=True,
                    )
                    endpoint_weight = config.get("endpoint_weight", 0.0)
                    if endpoint_weight:
                        endpoint = endpoint_l1(student, clean, cloudy, noise, config["sigma"]["maximum"])
                        loss = loss + endpoint_weight * endpoint
                        loss_components["endpoint_l1"] = endpoint.detach()
                        loss_components["weighted_endpoint_loss"] = (endpoint_weight * endpoint).detach()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at step {global_step + 1}")
            loss.backward()
            gradients = [p.grad for p in student.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError(f"missing or nonfinite gradient at step {global_step + 1}")
            gradient_norm = None
            gradient_clipped = False
            gradient_clip_norm = config.get("gradient_clip_norm")
            if gradient_clip_norm is not None:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    student.parameters(), gradient_clip_norm,
                ).item()
                if not math.isfinite(gradient_norm):
                    raise FloatingPointError(
                        f"nonfinite gradient norm at step {global_step + 1}"
                    )
                gradient_clipped = gradient_norm > gradient_clip_norm
            optimizer.step()
            update_ema(target, student, decay=config["ema_decay"])
            if inference_ema is not None:
                update_ema(inference_ema, student, decay=config['inference_ema_decay'])
            global_step += 1
            torch.cuda.synchronize()
            batch_size = clean.shape[0]
            value = loss.detach().item()
            epoch_loss += value * batch_size
            epoch_samples += batch_size
            record = {
                **input_audit,
                **{key: value.item() for key, value in loss_components.items()},
                "reconstruction_weight": config.get("reconstruction_weight", 0.0),
                "endpoint_weight": config.get("endpoint_weight", 0.0),
                "epoch": epoch, "batch": batch_index, "step": global_step,
                "batch_size": batch_size, "loss": value,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "sigma_high_min": high.min().item(),
                "sigma_high_max": high.max().item(),
                "gradient_norm_before_clip": gradient_norm,
                "gradient_clip_norm": gradient_clip_norm,
                "gradient_clipped": gradient_clipped,
                "allocated_mib": torch.cuda.memory_allocated() / 2**20,
                "reserved_mib": torch.cuda.memory_reserved() / 2**20,
                "step_seconds": time.monotonic() - step_started,
            }
            append_jsonl(metrics_path, record)
            if global_step == 1 or global_step % config["log_interval_steps"] == 0:
                print(json.dumps(record), flush=True)
        if epoch_samples != len(dataset):
            raise AssertionError("training epoch did not retain every sample")
        epoch_record = {
            "epoch": epoch, "global_step": global_step,
            "samples": epoch_samples, "mean_loss": epoch_loss / epoch_samples,
            "epoch_seconds": time.monotonic() - epoch_started,
        }
        append_jsonl(epochs_path, epoch_record)
        print(json.dumps({"epoch_summary": epoch_record}), flush=True)
        payload = checkpoint_payload(
            subset=args.subset, epoch=epoch, global_step=global_step,
            config=config, teacher_checkpoint=teacher_path,
            student=student, target=target, optimizer=optimizer,
            loader_generator=loader_generator, noise_generator=noise_generator,
        )
        if inference_ema is not None:
            payload['inference_ema'] = inference_ema.state_dict()
        save_rotating_checkpoint(output_dir, payload, keep_previous=not args.latest_only)
        if epoch in args.milestone_epochs:
            milestone = output_dir / f"epoch_{epoch:04d}.pt"
            if milestone.exists():
                raise FileExistsError(f"milestone already exists: {milestone}")
            shutil.copy2(output_dir / "latest.pt", milestone)
        del payload

    summary = {
        "status": "completed_to_requested_epoch",
        "subset": args.subset, "start_epoch": start_epoch,
        "end_epoch": args.target_epoch, "global_step": global_step,
        "steps_per_epoch": expected_steps, "samples_per_epoch": len(dataset),
        "teacher_checkpoint": str(teacher_path),
        "teacher_weights": config["teacher_weights"],
        "gradient_clip_norm": config.get("gradient_clip_norm"),
        "resume_checkpoint": str(args.resume.resolve()) if args.resume else None,
        "seconds": time.monotonic() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
