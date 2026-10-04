import ast
import copy
from pathlib import Path
import unittest


tree = ast.parse((Path(__file__).resolve().parents[1] / 'cloud_removal/train_teacher.py').read_text(encoding='utf-8'))
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_epoch_extension')
namespace = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), '<validator>', 'exec'), namespace)
validate = namespace['validate_epoch_extension']


class EpochExtensionTests(unittest.TestCase):
    def setUp(self):
        self.saved = dict(maximum_epochs=300, stage_epochs=[50, 150, 300],
                          optimizer={'learning_rate': 1e-4}, ema_decay=.999)
        self.current = copy.deepcopy(self.saved)
        self.current.update(maximum_epochs=400, stage_epochs=[50, 150, 300, 400])

    def test_extension(self):
        validate(self.saved, self.current)
        self.assertEqual(self.saved['maximum_epochs'], 300)

    def test_other_changes_rejected(self):
        for key, value in [('ema_decay', .99), ('optimizer', {'learning_rate': 1e-5})]:
            changed = {**self.current, key: value}
            with self.assertRaises(ValueError):
                validate(self.saved, changed)

    def test_stage_replacement_rejected(self):
        self.current['stage_epochs'] = [400]
        with self.assertRaises(ValueError):
            validate(self.saved, self.current)

    def test_no_extension_rejected(self):
        with self.assertRaises(ValueError):
            validate(self.saved, self.saved)


if __name__ == '__main__':
    unittest.main()
