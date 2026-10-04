"""Explicit teacher L1 extension; unrelated resume changes remain forbidden."""
import math


def reconstruction_weight(config):
    value = config['loss'].get('reconstruction_l1_weight', 0.0)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError('teacher L1 weight must be finite and nonnegative')
    return value


def validate_reconstruction_transition(saved, current):
    if reconstruction_weight(saved) != 0 or reconstruction_weight(current) != 1.0:
        raise ValueError('approved teacher transition is zero to fixed L1 weight 1.0')
    maximum = current['maximum_epochs']
    if maximum != saved['maximum_epochs'] + 10:
        raise ValueError('teacher reconstruction pilot must extend exactly ten epochs')
    expected = {**saved, 'maximum_epochs': maximum,
                'stage_epochs': [*saved['stage_epochs'], maximum],
                'loss': {**saved['loss'], 'reconstruction_l1_weight': 1.0}}
    if current != expected:
        raise ValueError('teacher L1 transition changed unrelated configuration')
