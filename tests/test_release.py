import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from cloud_removal.predict import build_package_model, predict_tensor
from scripts.export_inference import make_package


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(6, 3, 1)

    def forward(self, x, sigma):
        return self.projection(x)


class ReleaseTests(unittest.TestCase):
    def test_split_counts_and_relative_paths(self):
        root = Path(__file__).resolve().parents[1]
        for name, count in [('CUHK-CR1_train', 534), ('CUHK-CR1_test', 134),
                            ('CUHK-CR2_train', 448), ('CUHK-CR2_test', 111)]:
            rows = json.loads((root / f'data/manifests/{name}.json').read_text())
            self.assertEqual(len(rows), count)
            self.assertEqual(len({r['id'] for r in rows}), count)
            checks = json.loads((root / f'data/checksums/{name}.json').read_text())
            self.assertEqual([r['id'] for r in rows], [r['id'] for r in checks])
            for row in rows:
                for role in ('clean', 'cloudy'):
                    self.assertTrue(row[role].startswith('../raw/CUHK-CR/'))

    def test_export_roundtrip_student_without_teacher(self):
        from cloud_removal.edm import EDMInputScaling
        from cloud_removal.hdit_adapter import ConditionedHDiT
        from cloud_removal.training import ConsistencyFunction
        model = ConsistencyFunction(ConditionedHDiT(TinyBackbone(), input_transform=EDMInputScaling(.5)))
        state = {'kind': 'student_distillation_full_training_state', 'epoch': 20,
                 'subset': 'CUHK-CR1', 'target': model.state_dict(),
                 'config': {'sigma': {'minimum': .02, 'maximum': 20., 'data': .5},
                            'image_size': 8, 'student_input_scaling': 'edm'},
                 'teacher_checkpoint': 'PRIVATE_PATH', 'optimizer': {'private': 'unused'}}
        package = make_package(state, {}, 'a' * 64)
        self.assertNotIn('teacher_checkpoint', package)
        self.assertNotIn('optimizer', package)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'inference.pt'
            torch.save(package, path)
            package = torch.load(path, weights_only=True)
        with patch('cloud_removal.hdit_factory.build_hdit', return_value=TinyBackbone()):
            restored = build_package_model(package)
        cloudy = torch.randn(1, 3, 8, 8)
        self.assertTrue(torch.equal(predict_tensor(model, package, cloudy),
                                    predict_tensor(restored, package, cloudy)))

    def test_unknown_export_kind_rejected(self):
        with self.assertRaises(ValueError):
            make_package({'kind': 'unknown'}, {}, '')

    def test_invalid_package_rejected(self):
        with self.assertRaises(ValueError):
            build_package_model({'format': 'unknown'})
