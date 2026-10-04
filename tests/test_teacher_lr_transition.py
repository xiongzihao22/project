import copy
import unittest
from types import SimpleNamespace
from cloud_removal.teacher_lr_transition import validate_lr_transition, set_optimizer_lr


class LRTransitionTests(unittest.TestCase):
    def setUp(self):
        self.saved = {'optimizer': {'learning_rate': 1e-4, 'weight_decay': 0}, 'ema': .999}
        self.current = copy.deepcopy(self.saved)
        self.current['optimizer']['learning_rate'] = 3e-5

    def test_only_lr_allowed(self):
        validate_lr_transition(self.saved, self.current)
        self.assertEqual(self.saved['optimizer']['learning_rate'], 1e-4)
        self.current['ema'] = .9
        with self.assertRaises(ValueError):
            validate_lr_transition(self.saved, self.current)

    def test_invalid_rates(self):
        for value in (0, -1, float('nan'), float('inf'), True, 1e-4):
            self.current['optimizer']['learning_rate'] = value
            with self.assertRaises(ValueError):
                validate_lr_transition(self.saved, self.current)

    def test_all_groups_and_moments_preserved(self):
        moments = {'exp_avg': [1, 2]}
        optimizer = SimpleNamespace(param_groups=[{'lr': 1e-4}, {'lr': 1e-4}], state=moments)
        set_optimizer_lr(optimizer, 3e-5)
        self.assertEqual([g['lr'] for g in optimizer.param_groups], [3e-5, 3e-5])
        self.assertIs(optimizer.state, moments)
