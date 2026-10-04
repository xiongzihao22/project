"""Same-protocol full-test evaluation; no training and no server connection."""
import argparse
import importlib
import json
import time
import os
import hashlib
import gc
from pathlib import Path

os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')

from .evidence_protocol import SCHEMA_VERSION, digest, expected_nfe, validate_result, write_json


def teacher_weight_state(checkpoint, weights):
    if weights not in ('ema', 'online'):
        raise ValueError('unknown teacher weight selection')
    if weights == 'online' and checkpoint['kind'] != 'full_training_state':
        raise ValueError('online teacher weights require full training state')
    return checkpoint['ema_model' if weights == 'ema' else 'model']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=('teacher', 'student', 'direct', 'diffcr'), required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--subset', choices=('CUHK-CR1', 'CUHK-CR2'), required=True)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=700013)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--teacher-weights', choices=('ema', 'online'), default='ema')
    p.add_argument('--teacher-checkpoint', type=Path,
                   help='Student training source; defaults to its recorded path')
    p.add_argument('--expected-teacher-sha256',
                   help='Previously audited training-source hash, required for students')
    p.add_argument('--save-png', action='store_true')
    p.add_argument('--smoke-samples', type=int,
                   help='Diagnostic subset only; result cannot pass formal comparison')
    p.add_argument('--diffcr-factory', help='module:function returning the official adapter')
    args = p.parse_args()
    if args.method != 'teacher' and args.teacher_weights != 'ema':
        p.error('online weight selection is teacher-only')
    nfe = expected_nfe(args.method, args.steps)
    if args.batch_size < 1 or args.warmup < 1:
        p.error('batch-size and warmup must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    import lpips
    from torch.utils.data import DataLoader, Subset
    from .data import PairedRGBDataset, paired_model_transform
    from .image_metrics import mse_per_sample, psnr_per_sample, ssim_per_sample
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for the shared benchmark')
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / 'configs/data_protocol.json').read_text())
    split = protocol['subsets'][args.subset]
    manifest = root / split['test_manifest']
    dataset = PairedRGBDataset(manifest, pair_transform=paired_model_transform,
                               image_size=protocol['image_size'])
    if len(dataset) != split['test_count']:
        raise ValueError('fixed test split changed')
    if args.smoke_samples is not None and not 0 < args.smoke_samples < len(dataset):
        p.error('smoke-samples must be positive and smaller than the fixed test split')
    source_teacher_path = None
    checkpoint_hash = digest(args.checkpoint)
    if args.method == 'diffcr':
        if not args.diffcr_factory:
            p.error('DiffCR needs a verified official --diffcr-factory; no substitute sampler')
        module, name = args.diffcr_factory.split(':')
        adapter = getattr(importlib.import_module(module), name)(args.checkpoint, args.steps, args.subset)
        model, counted = adapter.model, adapter.denoiser
        predict = adapter.predict  # (cloudy, initial_noise) -> BCHW [-1,1]; never receives clean.
        provenance = adapter.provenance
        if not all(provenance.get(k) for k in ('commit', 'config_sha256', 'training_manifest_sha256', 'sampling_config', 'weights', 'conditioning')):
            raise ValueError('DiffCR provenance incomplete')
        if provenance['conditioning'] != 'single_rgb_cloudy':
            raise ValueError('multi-temporal conditioning is not the same input budget')
        if provenance['training_manifest_sha256'] != digest(root / split['train_manifest']):
            raise ValueError('DiffCR was not trained on the same manifest')
    else:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if checkpoint['subset'] != args.subset:
            raise ValueError('checkpoint subset mismatch')
        config = checkpoint['config']
        if config.get('image_size', 512) != protocol['image_size']:
            raise ValueError('checkpoint image size differs from protocol')
        provenance = {'epoch': checkpoint['epoch'], 'global_step': checkpoint['global_step'],
                      'config': config, 'checkpoint': str(args.checkpoint.resolve())}
        if args.method != 'direct':
            provenance['model_spec_sha256'] = digest(root / config['model_config'])
            provenance['model_spec'] = json.loads((root / config['model_config']).read_text())
        if args.method == 'teacher':
            from .train_teacher import build_model
            from .teacher_sampling import sample_conditional_edm
            if checkpoint['kind'] not in ('full_training_state', 'ema_model_only'):
                raise ValueError('expected teacher full state or EMA milestone')
            model = build_model(root, config)
            model.load_state_dict(teacher_weight_state(checkpoint, args.teacher_weights))
            provenance['weights'] = args.teacher_weights
            sampling = json.loads((root / 'configs/teacher_sampling.json').read_text())
            provenance['sampling_config'] = sampling
            predict = lambda cloudy, noise: sample_conditional_edm(model, cloudy, noise, args.steps, sampling)
        elif args.method == 'student':
            from .train_student import build_student
            from .train_student import validate_teacher_checkpoint
            from .student_ema import inference_ema_state
            if checkpoint['kind'] != 'student_distillation_full_training_state':
                raise ValueError('expected distilled student state')
            if not args.expected_teacher_sha256:
                p.error('student evidence requires --expected-teacher-sha256 from its training audit')
            source_teacher_path = args.teacher_checkpoint or Path(checkpoint['teacher_checkpoint'])
            source_teacher_hash = digest(source_teacher_path)
            if source_teacher_hash != args.expected_teacher_sha256:
                raise ValueError('student training-source teacher hash mismatch')
            source_teacher = torch.load(source_teacher_path, map_location='cpu', weights_only=False)
            validate_teacher_checkpoint(source_teacher, args.subset, config)
            provenance['teacher'] = {
                'checkpoint_sha256': source_teacher_hash,
                'epoch': source_teacher['epoch'], 'weights': config['teacher_weights'],
                'recorded_path': checkpoint['teacher_checkpoint'],
                'verified_path': str(source_teacher_path.resolve())}
            del source_teacher
            model = build_student(root, config)
            model.load_state_dict(inference_ema_state(checkpoint))
            def predict(cloudy, noise):
                sigma = torch.full((len(cloudy),), config['sigma']['maximum'], device='cuda')
                return model(noise * sigma[:, None, None, None], sigma, cloudy)
        else:
            from .direct_hdit import DirectHDiT
            if checkpoint['kind'] != 'direct_hdit_full_state':
                raise ValueError('expected independently trained direct regression')
            if config['train_manifest_sha256'] != digest(root / split['train_manifest']):
                raise ValueError('direct model training manifest changed')
            provenance['model_spec'] = checkpoint['model_spec']
            provenance['planned_training_epochs'] = config['epochs']
            provenance['completed_training_epochs'] = checkpoint['epoch']
            provenance['training_schedule_completed'] = checkpoint['epoch'] >= config['epochs']
            if checkpoint['epoch'] < config['epochs']:
                provenance['training_limitation'] = 'truncated_direct_baseline_not_converged_evidence'
            model = DirectHDiT(checkpoint['model_spec']).cuda()
            model.load_state_dict(checkpoint['ema_model'])
            predict = lambda cloudy, noise: model(cloudy)
        counted = model
        del checkpoint
        gc.collect()
    model.cuda().eval().requires_grad_(False)
    calls = [0]
    def count(module, inputs):
        calls[0] += 1
    hook = counted.register_forward_pre_hook(count)
    metric = lpips.LPIPS(net='alex', version='0.1').cuda().eval().requires_grad_(False)
    lpips_hash = hashlib.sha256()
    for key, value in sorted(metric.state_dict().items()):
        lpips_hash.update(key.encode())
        lpips_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    args.output.mkdir(parents=True, exist_ok=False)
    if args.save_png:
        (args.output / 'predictions').mkdir()
        from PIL import Image
        from .train_teacher_short import tensor_to_uint8
    rows, elapsed, batches, files = [], 0.0, [], []
    input_hash, noise_hash = hashlib.sha256(), hashlib.sha256()
    generator = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    evaluated = (Subset(dataset, range(args.smoke_samples))
                 if args.smoke_samples is not None else dataset)
    loader = DataLoader(evaluated, batch_size=args.batch_size, shuffle=False, num_workers=0)
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            for i, identity in enumerate(batch['id']):
                input_hash.update(identity.encode())
                for key in ('cloudy', 'clean'):
                    input_hash.update(batch[key][i].contiguous().numpy().tobytes())
            cloudy, clean = batch['cloudy'].cuda(), batch['clean'].cuda()
            noise_cpu = torch.randn(cloudy.shape, generator=generator)
            noise_hash.update(noise_cpu.contiguous().numpy().tobytes())
            noise = noise_cpu.cuda()
            if batch_index == 0:
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    for _ in range(args.warmup):
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            predict(cloudy, noise.clone())
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            calls[0] = 0
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = predict(cloudy, noise)
            torch.cuda.synchronize()
            batch_seconds = time.perf_counter() - started
            elapsed += batch_seconds
            if calls[0] != nfe:
                raise ValueError(f'actual NFE {calls[0]} != requested {nfe}')
            batches.append({'samples': len(cloudy), 'actual_nfe': calls[0],
                            'inference_seconds': batch_seconds})
            if prediction.shape != clean.shape or not torch.isfinite(prediction).all():
                raise ValueError('invalid prediction')
            prediction = prediction.float().clamp(-1, 1)
            values = {'psnr': psnr_per_sample(prediction, clean),
                      'ssim': ssim_per_sample(prediction, clean),
                      'rmse': mse_per_sample(prediction, clean).sqrt() / 2,
                      'lpips': metric(prediction, clean).flatten()}
            if not all(torch.isfinite(v).all() for v in values.values()):
                raise ValueError('nonfinite metric')
            for i, identity in enumerate(batch['id']):
                rows.append({'id': identity, **{k: float(v[i]) for k, v in values.items()}})
                if args.save_png:
                    relative = f'predictions/{len(rows) - 1:04d}.png'
                    path = args.output / relative
                    Image.fromarray(tensor_to_uint8(prediction[i])).save(path)
                    files.append({'path': relative, 'id': identity, 'sha256': digest(path)})
    hook.remove()
    if digest(args.checkpoint) != checkpoint_hash:
        raise ValueError('checkpoint changed during evaluation')
    if source_teacher_path is not None and digest(source_teacher_path) != source_teacher_hash:
        raise ValueError('student training-source teacher changed during evaluation')
    result = {'schema_version': SCHEMA_VERSION, 'status': 'completed',
              'scope': 'full_test' if args.smoke_samples is None else 'smoke_only',
              'method': args.method, 'weights': (args.teacher_weights if args.method == 'teacher'
                  else 'ema' if args.method != 'diffcr' else provenance['weights']),
              'checkpoint_sha256': checkpoint_hash, 'provenance': provenance,
              'manifest_sha256': digest(manifest), 'sample_ids': [r['id'] for r in rows],
              'train_manifest_sha256': digest(root / split['train_manifest']),
              'data_protocol_sha256': digest(root / 'configs/data_protocol.json'),
              'input_content_sha256': input_hash.hexdigest(),
              'initial_noise_sha256': noise_hash.hexdigest(),
              'metric_code_sha256': digest(root / 'cloud_removal/image_metrics.py'),
              'lpips_state_sha256': lpips_hash.hexdigest(),
              'test_split_size': len(dataset), 'evaluated_samples': len(rows),
              'metric_protocol': 'ssim_nonnegative_v2_reflect11_sigma1.5_rgb_mean',
              'lpips_protocol': 'alex_v0.1_float_clamped_minus1_plus1',
              'image_size': protocol['image_size'], 'batch_size': args.batch_size,
              'precision': 'bfloat16_autocast', 'device': torch.cuda.get_device_name(),
              'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
              'seed': args.seed, 'steps': args.steps, 'nfe': nfe,
              'warmup_batches': args.warmup, 'inference_seconds': elapsed,
              'timing_protocol': 'synchronized_wall_predict_only_bf16_v1', 'batches': batches,
              'seconds_per_image': elapsed / len(rows),
              'timing_excludes': ['data_loading', 'host_to_device', 'metrics', 'serialization'],
              'parameters': sum(v.numel() for v in model.parameters()),
              'metrics': {k: sum(r[k] for r in rows) / len(rows) for k in values},
              'per_image': rows, 'prediction_files': files}
    if args.smoke_samples is None:
        validate_result(result)
    write_json(args.output / 'results.json', result)
    print(json.dumps({k: result[k] for k in ('method', 'nfe', 'metrics', 'seconds_per_image')}))


if __name__ == '__main__':
    main()
