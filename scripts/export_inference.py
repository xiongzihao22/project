"""Export trusted full training states to path-free, EMA inference packages."""
import argparse
import json
from pathlib import Path

from cloud_removal.evidence_protocol import digest


def make_package(state, model_spec, source_hash):
    from cloud_removal.student_ema import inference_ema_state
    kinds = {'full_training_state': 'teacher', 'ema_model_only': 'teacher',
             'student_distillation_full_training_state': 'student',
             'direct_hdit_full_state': 'direct'}
    method = kinds.get(state.get('kind'))
    if method is None:
        raise ValueError('Unsupported checkpoint kind')
    config = state['config']
    if config.get('student_backbone', 'hdit') != 'hdit' or config.get('backbone_type', 'hdit') != 'hdit':
        raise ValueError('This export format currently supports HDiT only')
    weights = inference_ema_state(state) if method == 'student' else state['ema_model']
    return {'format': 'cloud_removal_inference_v1', 'method': method,
            'weights': 'ema', 'source_checkpoint_sha256': source_hash,
            'epoch': int(state['epoch']), 'subset': state['subset'],
            'image_size': int(config.get('image_size', 512)),
            'sigma': config.get('sigma', {}),
            'student_input_scaling': config.get('student_input_scaling', 'edm'),
            'model_spec': model_spec,
            'state_dict': {k: v.detach().cpu() for k, v in weights.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trust-checkpoint', action='store_true')
    args = parser.parse_args()
    if not args.trust_checkpoint:
        parser.error('Full training states require explicit --trust-checkpoint')
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    source_hash = digest(args.checkpoint)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    root = Path(__file__).resolve().parents[1]
    spec = (state['model_spec'] if state['kind'] == 'direct_hdit_full_state'
            else json.loads((root / state['config']['model_config']).read_text()))
    package = make_package(state, spec, source_hash)
    if digest(args.checkpoint) != source_hash:
        raise ValueError('Source changed during export')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        torch.save(package, stream)
    verified = torch.load(args.output, map_location='cpu', weights_only=True)
    if verified['source_checkpoint_sha256'] != source_hash:
        raise ValueError('Export validation failed')
    print(json.dumps({'method': package['method'], 'sha256': digest(args.output),
                      'source_checkpoint_sha256': source_hash}))


if __name__ == '__main__':
    main()
