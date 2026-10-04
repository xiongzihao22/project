"""Dependency-free contracts for the teacher/student evidence chain."""
import hashlib
import math
import json
import os
from pathlib import Path

SCHEMA_VERSION = 2
METRICS = ('psnr', 'ssim', 'lpips', 'rmse')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def expected_nfe(method, steps):
    if type(steps) is not int or steps < 1:
        raise ValueError('steps must be a positive integer')
    if method in ('student', 'direct'):
        if steps != 1:
            raise ValueError('one-pass models only support one step')
        return 1
    if method == 'teacher':
        if steps < 2:
            raise ValueError('Heun teacher requires at least two schedule points')
        return 2 * steps - 1
    if method == 'diffcr':
        return steps  # Requested NFE, verified by an actual denoiser hook.
    raise ValueError('unknown method')


def compare(reference, candidate, *, additional_teacher_baseline=False):
    for result in (reference, candidate):
        validate_result(result)
    for key in ('manifest_sha256', 'metric_protocol', 'lpips_protocol',
                'image_size', 'batch_size', 'precision', 'device', 'seed',
                'torch_version', 'cuda_version', 'input_content_sha256',
                'initial_noise_sha256', 'metric_code_sha256', 'lpips_state_sha256',
                'timing_protocol', 'train_manifest_sha256', 'data_protocol_sha256'):
        if reference[key] != candidate[key]:
            raise ValueError('incompatible comparison protocol: ' + key)
    if reference['sample_ids'] != candidate['sample_ids']:
        raise ValueError('test sample identities/order differ')
    methods = {reference['method'], candidate['method']}
    if methods == {'teacher', 'student'}:
        teacher = reference if reference['method'] == 'teacher' else candidate
        student = candidate if candidate['method'] == 'student' else reference
        source = student['provenance'].get('teacher', {})
        if (not additional_teacher_baseline and
                (source.get('checkpoint_sha256') != teacher['checkpoint_sha256']
                 or source.get('weights') != teacher['weights'])):
            raise ValueError('teacher is not the verified student training source')
    if methods == {'direct', 'student'}:
        if (reference['parameters'] != candidate['parameters']
                or reference['provenance']['model_spec'] != candidate['provenance']['model_spec']):
            raise ValueError('direct/student architecture capacity differs')
    return {
        'reference': {'method': reference['method'], 'checkpoint_sha256': reference['checkpoint_sha256']},
        'candidate': {'method': candidate['method'], 'checkpoint_sha256': candidate['checkpoint_sha256']},
        'training_limitations': {
            name: result['provenance'].get('training_limitation')
            for name, result in (('reference', reference), ('candidate', candidate))},
        'candidate_minus_reference': {
            key: candidate['metrics'][key] - reference['metrics'][key]
            for key in METRICS},
        'reference_over_candidate_latency': reference['inference_seconds'] / candidate['inference_seconds'],
        'reference_over_candidate_nfe': reference['nfe'] / candidate['nfe'],
        'equal_nfe': reference['nfe'] == candidate['nfe'],
        'comparison_scope': ('additional_teacher_baseline_not_student_distillation_source'
                             if additional_teacher_baseline else 'source_verified_comparison'),
        'interpretation': 'Candidate-minus-reference quality deltas and reference-over-candidate efficiency ratios',
    }


def validate_result(result):
    if result.get('schema_version') != SCHEMA_VERSION or result.get('status') != 'completed':
        raise ValueError('completed evidence schema v2 required')
    if result.get('scope') != 'full_test':
        raise ValueError('smoke results cannot be used for formal comparison')
    ids = result['sample_ids']
    if (not ids or len(ids) != len(set(ids))
            or len(ids) != result['test_split_size']
            or ids != [row['id'] for row in result['per_image']]):
        raise ValueError('incomplete or duplicate test identities')
    nfe = expected_nfe(result['method'], result['steps'])
    if result['nfe'] != nfe:
        raise ValueError('NFE does not match method and steps')
    batches = result['batches']
    if (not batches or sum(row['samples'] for row in batches) != len(ids)
            or any(row['samples'] <= 0 or row['actual_nfe'] != nfe for row in batches)):
        raise ValueError('actual NFE or batch coverage mismatch')
    durations = [row['inference_seconds'] for row in batches]
    if (any(not math.isfinite(t) or t <= 0 for t in durations)
            or not math.isfinite(result['inference_seconds'])
            or not math.isclose(sum(durations), result['inference_seconds'], rel_tol=1e-9)):
        raise ValueError('invalid inference timing')
    for key in METRICS:
        values = [row[key] for row in result['per_image']]
        if (not all(math.isfinite(v) for v in values)
                or not math.isfinite(result['metrics'][key])
                or not math.isclose(sum(values) / len(values), result['metrics'][key],
                                    rel_tol=1e-9, abs_tol=1e-12)):
            raise ValueError('nonfinite or inconsistent metric: ' + key)
