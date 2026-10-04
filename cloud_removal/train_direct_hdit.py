"""Explicit from-scratch cloudy-to-clean control; no teacher or test selection."""
import argparse
import copy
import json
import os
import shutil
import time
from pathlib import Path
from .evidence_protocol import digest, write_json

os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')


def budget_allows_next_epoch(elapsed, maximum, previous_epoch_seconds):
    return maximum is None or elapsed + previous_epoch_seconds < maximum


def checkpoint_epochs(values, maximum):
    if any(type(v) is not int or not 1 <= v <= maximum for v in values):
        raise ValueError('saved epochs must be integers within the training schedule')
    if len(values) != len(set(values)):
        raise ValueError('duplicate saved epochs')
    return sorted(values)


def retain_checkpoint(source, destination):
    if destination.exists():
        raise FileExistsError('retained checkpoint already exists')
    temporary = destination.with_suffix('.pt.tmp')
    with source.open('rb') as src, temporary.open('xb') as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    os.replace(temporary, destination)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subset', choices=('CUHK-CR1', 'CUHK-CR2'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, required=True)
    p.add_argument('--save-epochs', type=int, nargs='*', default=[])
    p.add_argument('--learning-rate', type=float, required=True)
    p.add_argument('--loss', choices=('l1', 'mse'), required=True)
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--max-seconds', type=float,
                   help='Optional wall-time guard checked at complete epoch boundaries')
    p.add_argument('--model-config', type=Path, default=Path('configs/hdit_candidate.json'))
    args = p.parse_args()
    import math
    if (args.epochs < 1 or not math.isfinite(args.learning_rate)
            or args.learning_rate <= 0 or args.num_workers < 0):
        p.error('positive epochs and learning rate required')
    if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
        p.error('max-seconds must be finite and positive')
    try:
        saved_epochs = checkpoint_epochs(args.save_epochs, args.epochs)
    except ValueError as error:
        p.error(str(error))
    import torch
    import k_diffusion as K
    from torch.utils.data import DataLoader
    from .data import PairedRGBDataset, paired_model_transform
    from .direct_hdit import DirectHDiT
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / args.model_config).read_text())
    protocol = json.loads((root / 'configs/data_protocol.json').read_text())
    split = protocol['subsets'][args.subset]
    manifest = root / split['train_manifest']
    dataset = PairedRGBDataset(manifest, pair_transform=paired_model_transform,
                               image_size=protocol['image_size'])
    if len(dataset) != split['train_count']:
        raise ValueError('fixed training split changed')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = DirectHDiT(spec).cuda()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  betas=(0.9, 0.999), weight_decay=0)
    generator = torch.Generator().manual_seed(args.seed + 1)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, generator=generator,
                        num_workers=args.num_workers, drop_last=False, pin_memory=True,
                        persistent_workers=args.num_workers > 0)
    config = {'loss': args.loss, 'learning_rate': args.learning_rate, 'seed': args.seed,
              'epochs': args.epochs, 'batch_size': 4, 'ema_decay': 0.999,
              'initialization': 'random_no_teacher', 'constant_time': 1.0,
              'input': 'concat(zero,cloudy)', 'precision': 'bfloat16',
              'image_size': protocol['image_size'], 'num_workers': args.num_workers,
              'max_seconds': args.max_seconds,
              'model_spec_sha256': digest(root / args.model_config),
              'train_manifest_sha256': digest(manifest), 'gradient_clip': 0.01,
              'status': 'experimental_direct_regression'}
    write_json(args.output / 'config.json', config)
    write_json(args.output / 'retention.json', {'epochs': saved_epochs})
    loss_fn = torch.nn.functional.l1_loss if args.loss == 'l1' else torch.nn.functional.mse_loss
    step = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.monotonic()
        seen = 0
        total_loss = 0.0
        model.train()
        for batch_index, batch in enumerate(loader, start=1):
            cloudy = batch['cloudy'].cuda(non_blocking=True)
            clean = batch['clean'].cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with K.models.checkpointing(True), torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(cloudy)
                loss = loss_fn(prediction.float(), clean)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for target, source in zip(ema.parameters(), model.parameters()):
                    target.lerp_(source, 0.001)
                for target, source in zip(ema.buffers(), model.buffers()):
                    target.copy_(source)
            step += 1
            seen += len(cloudy)
            total_loss += float(loss) * len(cloudy)
            with (args.output / 'training_metrics.jsonl').open('a') as stream:
                stream.write(json.dumps({'epoch': epoch, 'step': step, 'loss': float(loss),
                                          'batch': batch_index, 'batches': len(loader),
                                          'learning_rate': optimizer.param_groups[0]['lr'],
                                          'gradient_norm': float(norm), 'ids': batch['id']}) + '\n')
        if seen != split['train_count']:
            raise ValueError('training epoch omitted original split samples')
        state = {'kind': 'direct_hdit_full_state', 'subset': args.subset,
                 'epoch': epoch, 'global_step': step, 'config': config, 'model_spec': spec,
                 'model': model.state_dict(), 'ema_model': ema.state_dict(),
                 'optimizer': optimizer.state_dict(), 'rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all(), 'loader_rng': generator.get_state()}
        temporary = args.output / 'latest.pt.tmp'
        torch.save(state, temporary)
        os.replace(temporary, args.output / 'latest.pt')
        if epoch in saved_epochs:
            retain_checkpoint(args.output / 'latest.pt', args.output / f'epoch{epoch}.pt')
        torch.cuda.synchronize()
        summary = {'status': 'completed' if epoch == args.epochs else 'running',
                   'epoch': epoch, 'global_step': step, 'target_epoch': args.epochs,
                   'completed_epochs': epoch, 'planned_epochs': args.epochs,
                   'stop_reason': 'target_epochs_reached' if epoch == args.epochs else None,
                   'samples_per_epoch': seen, 'steps_per_epoch': len(loader),
                   'mean_training_loss': total_loss / seen,
                   'epoch_seconds': time.monotonic() - epoch_started,
                   'training_seconds': time.monotonic() - started,
                   'training_seconds_include': ['data_loading', 'training', 'checkpoint_save'],
                   'subset': args.subset, 'config': config,
                   'parameters': sum(v.numel() for v in model.parameters())}
        if epoch < args.epochs and not budget_allows_next_epoch(
                summary['training_seconds'], args.max_seconds, summary['epoch_seconds']):
            summary['status'] = 'budget_stopped'
            summary['stop_reason'] = 'wall_time_guard_at_completed_epoch'
        write_json(args.output / 'summary.json', summary)
        print(json.dumps(summary), flush=True)
        if summary['status'] == 'budget_stopped':
            break


if __name__ == '__main__':
    main()
