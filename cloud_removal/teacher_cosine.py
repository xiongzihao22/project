"""Global-step cosine decay; a short run does not compress the full schedule."""
import math


def validate_schedule(config):
    schedule = config['learning_rate_schedule']
    if schedule == 'constant':
        return
    if not isinstance(schedule, dict) or set(schedule) != {'type', 'start_step', 'end_step', 'minimum_lr'}:
        raise ValueError('invalid cosine schedule fields')
    if schedule['type'] != 'cosine':
        raise ValueError('unsupported schedule')
    start, end = schedule['start_step'], schedule['end_step']
    if any(type(x) is not int for x in (start, end)) or not 0 <= start < end:
        raise ValueError('invalid cosine step interval')
    lo, hi = schedule['minimum_lr'], config['optimizer']['learning_rate']
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (lo, hi)) or not 0 < lo < hi:
        raise ValueError('invalid cosine rates')


def learning_rate_at(config, completed_steps):
    validate_schedule(config)
    hi = config['optimizer']['learning_rate']
    schedule = config['learning_rate_schedule']
    if schedule == 'constant':
        return hi
    fraction = min(1., max(0., (completed_steps - schedule['start_step']) /
                           (schedule['end_step'] - schedule['start_step'])))
    lo = schedule['minimum_lr']
    return lo + (hi - lo) * (1. + math.cos(math.pi * fraction)) / 2.


def validate_cosine_transition(saved, current, source_step):
    validate_schedule(current)
    schedule = current['learning_rate_schedule']
    if saved['learning_rate_schedule'] != 'constant' or not isinstance(schedule, dict):
        raise ValueError('transition requires constant to cosine')
    if schedule['start_step'] != source_step:
        raise ValueError('schedule must start at the source checkpoint')
    if current['maximum_epochs'] <= saved['maximum_epochs']:
        raise ValueError('schedule transition must extend epoch limit')
    expected = {**saved, 'maximum_epochs': current['maximum_epochs'],
                'stage_epochs': [*saved['stage_epochs'], current['maximum_epochs']],
                'learning_rate_schedule': schedule}
    if current != expected:
        raise ValueError('cosine transition changed unrelated settings')
