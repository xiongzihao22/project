import copy
import unittest
from cloud_removal.teacher_stratified import sampling_mode, validate_stratified_transition


class TransitionTests(unittest.TestCase):
    def setUp(self):
        self.saved = dict(maximum_epochs=800, stage_epochs=[700, 800], seed=13,
                          loss={'reduction': 'pixel_mean', 'weighting': 'edm'},
                          ema_decay=.999, sigma={'maximum': 20, 'minimum': .02, 'data': .5},
                          optimizer={'learning_rate': 3e-5}, learning_rate_schedule='constant')
        self.current = copy.deepcopy(self.saved)
        self.current.update(maximum_epochs=810, stage_epochs=[700, 800, 810],
                            sigma_sampling='stratified_lognormal')

    def test_approved(self):
        validate_stratified_transition(self.saved, self.current)
        self.assertEqual(sampling_mode(self.saved), 'independent')

    def test_unrelated_changes_rejected(self):
        for key, value in [('ema_decay', .99), ('sigma', {'maximum': 10}),
                           ('optimizer', {'learning_rate': 1e-4}), ('seed', 14),
                           ('learning_rate_schedule', 'cosine'), ('stage_epochs', [810]),
                           ('maximum_epochs', 820),
                           ('loss', {'reduction': 'pixel_mean', 'weighting': 'soft_min_snr'}),
                           ('loss', {**self.saved['loss'], 'reconstruction_l1_weight': 1.})]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_stratified_transition(self.saved, {**self.current, key: value})

    def test_wrong_mode_and_mixture_rejected(self):
        with self.assertRaises(ValueError):
            sampling_mode({**self.saved, 'sigma_sampling': 'unknown'})
        c = copy.deepcopy(self.current)
        c['sigma']['high_noise_mixture'] = {'probability': .25}
        with self.assertRaises(ValueError):
            validate_stratified_transition(self.saved, c)


if __name__ == '__main__':
    unittest.main()
