import copy
import unittest

from cloud_removal.teacher_reconstruction import reconstruction_weight, validate_reconstruction_transition


class TransitionTests(unittest.TestCase):
    def setUp(self):
        self.saved = dict(maximum_epochs=800, stage_epochs=[700, 800],
                          loss={'reduction': 'pixel_mean', 'weighting': 'edm'},
                          ema_decay=.999, sigma={'maximum': 20},
                          optimizer={'learning_rate': 3e-5}, learning_rate_schedule='constant')
        self.current = copy.deepcopy(self.saved)
        self.current.update(maximum_epochs=810, stage_epochs=[700, 800, 810])
        self.current['loss']['reconstruction_l1_weight'] = 1.0

    def test_approved(self):
        validate_reconstruction_transition(self.saved, self.current)
        self.assertEqual(reconstruction_weight(self.saved), 0)

    def test_unrelated_changes(self):
        for key, value in [('ema_decay', .99), ('sigma', {'maximum': 10}),
                           ('optimizer', {'learning_rate': 1e-4}),
                           ('learning_rate_schedule', 'cosine'), ('stage_epochs', [810])]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_reconstruction_transition(self.saved, {**self.current, key: value})

    def test_reject_wrong_extension(self):
        with self.assertRaises(ValueError):
            validate_reconstruction_transition(self.saved, {**self.current, 'maximum_epochs': 820})

    def test_reject_weights(self):
        for value in [True, -1, float('nan'), float('inf')]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                reconstruction_weight({'loss': {'reconstruction_l1_weight': value}})
        for value in [0, .1, 2]:
            c = copy.deepcopy(self.current)
            c['loss']['reconstruction_l1_weight'] = value
            with self.assertRaises(ValueError):
                validate_reconstruction_transition(self.saved, c)


if __name__ == '__main__':
    unittest.main()
