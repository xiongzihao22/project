"""Explicit learning-rate-only transitions preserving optimizer moments."""
import copy
import math


def validate_lr_transition(saved, current):
    rate = current['optimizer']['learning_rate']
    if isinstance(rate, bool) or not math.isfinite(rate) or rate <= 0:
        raise ValueError('learning rate must be finite and positive')
    expected = copy.deepcopy(saved)
    expected['optimizer']['learning_rate'] = rate
    if expected != current:
        raise ValueError('learning-rate transition must not alter any other setting')
    if rate == saved['optimizer']['learning_rate']:
        raise ValueError('no learning-rate transition requested')


def set_optimizer_lr(optimizer, rate):
    for group in optimizer.param_groups:
        group['lr'] = rate
