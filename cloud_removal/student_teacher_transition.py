"""Narrow opt-in for the approved epoch-100 to epoch-110 teacher switch."""
import copy


def normalized_resume_config(saved, current, enabled):
    result = copy.deepcopy(current)
    if not enabled:
        return result
    if (saved.get('teacher_epoch'), current.get('teacher_epoch')) != (100, 110):
        raise ValueError('only the approved teacher 100-to-110 transition is allowed')
    if saved.get('teacher_weights') != 'ema' or current.get('teacher_weights') != 'ema':
        raise ValueError('teacher transition requires EMA weights')
    expected = copy.deepcopy(saved)
    expected['teacher_epoch'] = 110
    expected['maximum_epochs'] = current['maximum_epochs']
    if current != expected or current['maximum_epochs'] < saved['maximum_epochs']:
        raise ValueError('teacher transition cannot change other training settings')
    result['teacher_epoch'] = 100
    return result
