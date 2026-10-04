import copy
import unittest
from cloud_removal.teacher_lr_reheat import validate_reheat_transition


class TransitionTests(unittest.TestCase):
    def setUp(self):
        self.saved = dict(maximum_epochs=800, stage_epochs=[720, 800], seed=13,
            loss={'reduction': 'pixel_mean', 'weighting': 'edm'}, ema_decay=.999,
            sigma={'minimum': .02, 'maximum': 20, 'data': .5},
            optimizer={'learning_rate': 3e-5, 'betas': [.9, .999]},
            learning_rate_schedule=dict(type='cosine', start_step=96480, end_step=107200, minimum_lr=3e-6))

    def config(self, lr):
        c = copy.deepcopy(self.saved)
        c.update(maximum_epochs=810, stage_epochs=[720, 800, 810], learning_rate_schedule='constant')
        c['optimizer']['learning_rate'] = lr
        return c

    def test_approved_rates(self):
        for lr in (3e-6, 1e-5):
            validate_reheat_transition(self.saved, self.config(lr), 107200)

    def test_reject_unrelated(self):
        for key, value in [('ema_decay', .99), ('seed', 14), ('sigma', {'maximum': 10}),
            ('stage_epochs', [810]), ('maximum_epochs', 820), ('sigma_sampling', 'stratified_lognormal'),
            ('loss', {'reduction': 'pixel_mean', 'weighting': 'soft_min_snr'})]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_reheat_transition(self.saved, {**self.config(1e-5), key: value}, 107200)
        c = self.config(1e-5)
        c['optimizer']['betas'] = [.8, .99]
        with self.assertRaises(ValueError):
            validate_reheat_transition(self.saved, c, 107200)

    def test_reject_rate_and_early_source(self):
        for lr in (0, 1e-4, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                validate_reheat_transition(self.saved, self.config(lr), 107200)
        with self.assertRaises(ValueError):
            validate_reheat_transition(self.saved, self.config(1e-5), 107199)


if __name__ == '__main__':
    unittest.main()
