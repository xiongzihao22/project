"""Cloudy-only inference with an exported EMA package; no teacher file required."""
import argparse
from contextlib import nullcontext
from pathlib import Path


def build_package_model(package):
    from .hdit_factory import build_hdit
    from .hdit_adapter import ConditionedHDiT
    from .edm import EDMInputScaling, EDMTeacher, identity_inputs
    from .training import ConsistencyFunction
    from .direct_hdit import DirectHDiT
    if package.get('format') != 'cloud_removal_inference_v1' or package.get('weights') != 'ema':
        raise ValueError('Expected an exported EMA inference package')
    method = package['method']
    if method == 'direct':
        model = DirectHDiT(package['model_spec'])
    elif method in ('teacher', 'student'):
        sigma = package['sigma']
        scaling = package['student_input_scaling'] if method == 'student' else 'edm'
        if scaling not in ('edm', 'identity'):
            raise ValueError('Unknown input scaling')
        transform = EDMInputScaling(sigma['data']) if scaling == 'edm' else identity_inputs
        backbone = ConditionedHDiT(build_hdit(package['model_spec']), input_transform=transform)
        model = (EDMTeacher(backbone, sigma_data=sigma['data']) if method == 'teacher'
                 else ConsistencyFunction(backbone, sigma_min=sigma['minimum'], sigma_data=sigma['data']))
    else:
        raise ValueError('Unsupported method')
    model.load_state_dict(package['state_dict'], strict=True)
    return model.eval().requires_grad_(False)


def predict_tensor(model, package, cloudy, seed=700013, teacher_steps=18):
    import torch
    from .teacher_sampling import sample_conditional_edm
    if package['method'] == 'direct':
        return model(cloudy)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    noise = torch.randn(cloudy.shape, generator=generator).to(cloudy.device)
    sigma = package['sigma']
    if package['method'] == 'student':
        levels = torch.full((len(cloudy),), sigma['maximum'], device=cloudy.device)
        return model(noise * levels[:, None, None, None], levels, cloudy)
    import json
    sampling = json.loads((Path(__file__).resolve().parents[1] / 'configs/teacher_sampling.json').read_text())
    return sample_conditional_edm(model, cloudy, noise, teacher_steps, sampling)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--seed', type=int, default=700013)
    parser.add_argument('--teacher-steps', type=int, default=18)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from PIL import Image
    from .data import paired_model_transform
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; CPU is only suitable for supported attention backends')
    package = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    with Image.open(args.input) as source:
        if source.mode != 'RGB' or source.size != (package['image_size'],) * 2:
            raise ValueError('Expected RGB image at the checkpoint resolution; no implicit resize')
        cloudy, _ = paired_model_transform(source, source)
    model = build_package_model(package).to(args.device)
    cloudy = cloudy.unsqueeze(0).to(args.device)
    autocast = torch.autocast('cuda', dtype=torch.bfloat16) if args.device == 'cuda' else nullcontext()
    with torch.inference_mode(), autocast:
        prediction = predict_tensor(model, package, cloudy, args.seed, args.teacher_steps)
    if prediction.shape != cloudy.shape or not torch.isfinite(prediction).all():
        raise ValueError('Invalid prediction')
    image = ((prediction[0].float().clamp(-1, 1) + 1) * 127.5).round().byte()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        Image.fromarray(image.permute(1, 2, 0).cpu().numpy()).save(stream, format='PNG')


if __name__ == '__main__':
    main()
