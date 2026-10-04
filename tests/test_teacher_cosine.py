import copy
import unittest
from cloud_removal.teacher_cosine import learning_rate_at, validate_cosine_transition, validate_schedule


class CosineTests(unittest.TestCase):
    def setUp(self):
        self.old = dict(maximum_epochs=720, stage_epochs=[700, 720],
                        learning_rate_schedule='constant', optimizer={'learning_rate': 3e-5},
                        ema_decay=.999, sigma={'maximum': 20})
        self.new = {**self.old, 'maximum_epochs': 800, 'stage_epochs': [700, 720, 800],
                    'learning_rate_schedule': dict(type='cosine', start_step=96480,
                                                   end_step=107200, minimum_lr=3e-6)}

    def test_endpoints_and_short_stop(self):
        self.assertAlmostEqual(learning_rate_at(self.new, 96480), 3e-5)
        self.assertAlmostEqual(learning_rate_at(self.new, 107200), 3e-6)
        self.assertAlmostEqual(learning_rate_at(self.new, 97820), 2.897237368890237e-5)
        self.assertEqual(learning_rate_at(self.old, 97820), 3e-5)

    def test_resume_keeps_global_curve(self):
        whole = [learning_rate_at(self.new, x) for x in range(96480, 97820)]
        resumed = [learning_rate_at(copy.deepcopy(self.new), x) for x in range(97150, 97820)]
        self.assertEqual(whole[670:], resumed)
        self.assertTrue(all(a >= b for a, b in zip(whole, whole[1:])))

    def test_transition_only_schedule_and_extension(self):
        validate_cosine_transition(self.old, self.new, 96480)
        for key, value in [('ema_decay', .99), ('sigma', {'maximum': 30}),
                           ('optimizer', {'learning_rate': 1e-5})]:
            bad = copy.deepcopy(self.new); bad[key] = value
            with self.assertRaises(ValueError): validate_cosine_transition(self.old, bad, 96480)
        with self.assertRaises(ValueError): validate_cosine_transition(self.old, self.new, 96481)

    def test_invalid_schedule(self):
        for key, value in [('end_step', 96480), ('minimum_lr', float('nan')),
                           ('minimum_lr', 0), ('minimum_lr', 4e-5), ('type', 'other')]:
            bad = copy.deepcopy(self.new); bad['learning_rate_schedule'][key] = value
            with self.assertRaises(ValueError): validate_schedule(bad)


if __name__ == '__main__':
    unittest.main()
