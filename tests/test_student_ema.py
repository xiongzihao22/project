import copy
import unittest

from cloud_removal.student_ema import (
    inference_ema_state, normalize_ema_transition, validate_ema_config,
)


class StudentEMATests(unittest.TestCase):
    def setUp(self):
        self.saved = dict(ema_decay=.999, maximum_epochs=20, learning_rate=1e-5)

    def test_approved_migrations(self):
        for decay in (.999, .99, .95):
            requested = dict(self.saved, ema_decay=decay,
                             inference_ema_decay=.999, maximum_epochs=25)
            result = normalize_ema_transition(self.saved, requested, True)
            self.assertEqual(result, dict(self.saved, maximum_epochs=25))
            self.assertEqual(requested['ema_decay'], decay)

    def test_unrelated_changes_rejected(self):
        requested = dict(self.saved, inference_ema_decay=.999, maximum_epochs=25)
        for key, value in [('learning_rate', 1e-4), ('maximum_epochs', 19),
                           ('ema_decay', .9), ('inference_ema_decay', .99)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                normalize_ema_transition(self.saved, dict(requested, **{key: value}), True)

    def test_no_permission_does_not_normalize(self):
        requested = dict(self.saved, inference_ema_decay=.999)
        self.assertEqual(normalize_ema_transition(self.saved, requested, False), requested)

    def test_invalid_decay(self):
        for value in (True, float('nan'), float('inf'), -1, 1, '0.999'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_ema_config(dict(ema_decay=value))

    def test_legacy_and_dual_selection(self):
        legacy = dict(config=self.saved, target={'weight': 1})
        self.assertIs(inference_ema_state(legacy), legacy['target'])
        dual = copy.deepcopy(legacy)
        dual['config']['inference_ema_decay'] = .999
        with self.assertRaises(ValueError):
            inference_ema_state(dual)
        dual['inference_ema'] = {'weight': 2}
        self.assertIs(inference_ema_state(dual), dual['inference_ema'])
        self.assertNotEqual(inference_ema_state(dual), dual['target'])

    def test_native_dual_resume(self):
        saved = dict(self.saved, inference_ema_decay=.999)
        requested = dict(saved, maximum_epochs=25)
        self.assertEqual(normalize_ema_transition(saved, requested, False), requested)
        with self.assertRaises(ValueError):
            normalize_ema_transition(saved, requested, True)


class StudentEMAUpdateTests(unittest.TestCase):
    def test_dual_control_matches_legacy_and_roundtrip(self):
        import io
        import torch
        from cloud_removal.training import make_target, update_ema
        torch.manual_seed(42)
        online = torch.nn.Linear(4, 3)
        legacy, target, inference = [make_target(online) for _ in range(3)]
        optimizer = torch.optim.AdamW(online.parameters(), lr=1e-5)
        for _ in range(5):
            optimizer.zero_grad()
            online(torch.randn(2, 4)).square().mean().backward()
            optimizer.step()
            for model in (legacy, target, inference):
                update_ema(model, online, decay=.999)
        for a, b, c in zip(legacy.parameters(), target.parameters(), inference.parameters()):
            self.assertTrue(torch.equal(a, b) and torch.equal(a, c))
            self.assertNotEqual(b.data_ptr(), c.data_ptr())
        update_ema(target, online, decay=.95)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(target.parameters(), inference.parameters())))
        payload = dict(config={'inference_ema_decay': .999}, target=target.state_dict(),
                       inference_ema=inference.state_dict(), optimizer=optimizer.state_dict())
        stream = io.BytesIO()
        torch.save(payload, stream)
        stream.seek(0)
        restored = torch.load(stream)
        self.assertTrue(all(torch.equal(v, inference_ema_state(restored)[k])
                            for k, v in inference.state_dict().items()))


if __name__ == '__main__':
    unittest.main()
