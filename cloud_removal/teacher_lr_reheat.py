"""Strict post-cosine ten-epoch LR pilot and RNG-neutral update diagnostics."""
import copy
import math

from .teacher_cosine import validate_schedule


def validate_reheat_transition(saved, current, source_step):
    validate_schedule(saved)
    schedule = saved['learning_rate_schedule']
    if not isinstance(schedule, dict) or source_step < schedule['end_step']:
        raise ValueError('LR pilot requires a completed cosine schedule')
    if saved['loss'] != {'reduction': 'pixel_mean', 'weighting': 'edm'}:
        raise ValueError('LR pilot requires original EDM only')
    if saved.get('sigma_sampling', 'independent') != 'independent' or 'high_noise_mixture' in saved['sigma']:
        raise ValueError('LR pilot requires original independent sampling')
    lr = current['optimizer']['learning_rate']
    if isinstance(lr, bool) or lr not in (3e-6, 1e-5) or schedule['minimum_lr'] != 3e-6:
        raise ValueError('unapproved LR pilot rate')
    expected = copy.deepcopy(saved)
    expected.update(maximum_epochs=saved['maximum_epochs'] + 10,
                    stage_epochs=[*saved['stage_epochs'], saved['maximum_epochs'] + 10],
                    learning_rate_schedule='constant')
    expected['optimizer']['learning_rate'] = lr
    if current != expected:
        raise ValueError('LR pilot changed unrelated settings')
    validate_schedule(current)


def parameter_snapshot(model):
    return {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}


def update_statistics(model, before):
    import torch
    with torch.no_grad():
        parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
        if parameters.keys() != before.keys():
            raise ValueError('parameter identity changed')
        delta2 = torch.stack([(parameters[n].float() - p.float()).square().sum()
                              for n, p in before.items()]).sum().item()
        norm2 = torch.stack([p.float().square().sum() for p in before.values()]).sum().item()
    if not math.isfinite(delta2) or not math.isfinite(norm2) or norm2 <= 0:
        raise FloatingPointError('invalid parameter update statistics')
    return dict(update_l2=math.sqrt(delta2), parameter_l2_before=math.sqrt(norm2),
                relative_update_l2=math.sqrt(delta2 / norm2))
