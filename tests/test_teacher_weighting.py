import copy
import unittest

from cloud_removal.teacher_weighting import weighting_mode, validate_soft_min_snr_transition


class TransitionTests(unittest.TestCase):
    def setUp(self):
        self.saved = dict(maximum_epochs=800, stage_epochs=[700, 800],
                          loss={'reduction': 'pixel_mean', 'weighting': 'edm'},
                          ema_decay=.999, sigma={'maximum': 20, 'data': .5},
                          optimizer={'learning_rate': 3e-5}, learning_rate_schedule='constant')
        self.current = copy.deepcopy(self.saved)
        self.current.update(maximum_epochs=810, stage_epochs=[700, 800, 810])
        self.current['loss']['weighting'] = 'soft_min_snr'

    def test_approved(self):
        validate_soft_min_snr_transition(self.saved, self.current)

    def test_unrelated_changes(self):
        for key, value in [('ema_decay', .99), ('sigma', {'maximum': 10}),
                           ('optimizer', {'learning_rate': 1e-4}),
                           ('learning_rate_schedule', 'cosine'), ('stage_epochs', [810]),
                           ('maximum_epochs', 820)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_soft_min_snr_transition(self.saved, {**self.current, key: value})

    def test_reject_additional_loss_or_wrong_mode(self):
        for loss in [{**self.current['loss'], 'reconstruction_l1_weight': 1.},
                     {**self.current['loss'], 'reduction': 'l2'},
                     {**self.current['loss'], 'weighting': 'edm'}]:
            with self.subTest(loss=loss), self.assertRaises(ValueError):
                validate_soft_min_snr_transition(self.saved, {**self.current, 'loss': loss})
        with self.assertRaises(ValueError):
            weighting_mode({'loss': {'weighting': 'unknown'}})


if __name__ == '__main__':
    unittest.main()
