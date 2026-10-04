"""Separate training-target and inference EMA, with legacy checkpoint support."""
import copy
import math


def validate_ema_config(config):
    for key in ('ema_decay', 'inference_ema_decay'):
        if key not in config:
            continue
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError('invalid ' + key)


def normalize_ema_transition(saved, requested, enabled):
    validate_ema_config(saved)
    validate_ema_config(requested)
    result = copy.deepcopy(requested)
    if not enabled:
        return result
    if 'inference_ema_decay' in saved:
        raise ValueError('EMA split migration requires a legacy checkpoint')
    if saved['ema_decay'] != .999 or requested.get('inference_ema_decay') != .999:
        raise ValueError('approved inference EMA must remain .999')
    if requested['ema_decay'] not in (.999, .99, .95):
        raise ValueError('unapproved target EMA')
    result.pop('inference_ema_decay')
    result['ema_decay'] = saved['ema_decay']
    expected = copy.deepcopy(saved)
    expected['maximum_epochs'] = requested['maximum_epochs']
    if result != expected or requested['maximum_epochs'] < saved['maximum_epochs']:
        raise ValueError('EMA migration cannot change other settings')
    return result


def inference_ema_state(checkpoint):
    if 'inference_ema_decay' in checkpoint['config']:
        if 'inference_ema' not in checkpoint:
            raise ValueError('dual-EMA checkpoint lacks inference weights')
        return checkpoint['inference_ema']
    return checkpoint['target']
