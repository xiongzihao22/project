import copy
import types
import unittest
from unittest.mock import patch
from cloud_removal.hdit_factory import build_hdit, validate_spec


class FactoryTests(unittest.TestCase):
    def setUp(self):
        # Interface fixture only, not a proposed experiment architecture.
        self.spec = {'patch_size': [2, 2],
                     'mapping': {'depth': 1, 'width': 32, 'd_ff': 96, 'dropout': 0.},
                     'levels': [{'depth': 1, 'width': 32, 'd_ff': 96, 'dropout': 0.,
                                 'attention': {'type': 'global', 'd_head': 8}}]}

    def test_explicit_channel_contract(self):
        fake = types.SimpleNamespace(GlobalAttentionSpec=lambda *x: x,
            LevelSpec=lambda *x: x, MappingSpec=lambda *x: x,
            ImageTransformerDenoiserModelV2=lambda **kw: kw)
        before = copy.deepcopy(self.spec)
        with patch('cloud_removal.hdit_factory.importlib.import_module', return_value=fake):
            model = build_hdit(self.spec)
        self.assertEqual((model['in_channels'], model['out_channels']), (6, 3))
        self.assertEqual(self.spec, before)

    def test_missing_parameter_rejected(self):
        del self.spec['mapping']['dropout']
        with self.assertRaises(ValueError):
            validate_spec(self.spec)

    def test_attention_not_silently_replaced(self):
        self.spec['levels'][0]['attention'] = {'type': 'neighborhood', 'd_head': 8, 'kernel_size': 7}
        with patch('cloud_removal.hdit_factory.importlib.import_module', return_value=types.SimpleNamespace(natten=None)):
            with self.assertRaisesRegex(RuntimeError, 'NATTEN'):
                build_hdit(self.spec)

    def test_invalid_heads_rejected(self):
        self.spec['levels'][0]['attention']['d_head'] = 7
        with self.assertRaises(ValueError):
            validate_spec(self.spec)
