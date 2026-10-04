import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from cloud_removal.data import PairedRGBDataset, paired_model_transform, resolve_image_path


def test_transform(clean, cloudy):
    # A shape-only fixture; not a proposed image normalization.
    return torch.zeros(3, 8, 8), torch.ones(3, 8, 8)


class DataTests(unittest.TestCase):
    @patch.dict('cloud_removal.data.os.environ', {}, clear=True)
    @patch('cloud_removal.data.platform.release', return_value='6.18.33.2-microsoft-standard-WSL2')
    def test_windows_absolute_path_resolves_on_wsl_kernel(self, _release):
        path = resolve_image_path(Path('/tmp/manifests'), r'D:\dataset\clean\1.png')
        self.assertEqual(path, Path('/mnt/d/dataset/clean/1.png'))

    def test_model_range_and_channels(self):
        clean = Image.new('RGB', (8, 8), (0, 128, 255))
        cloudy = Image.new('RGB', (8, 8), (255, 0, 128))
        a, b = paired_model_transform(clean, cloudy)
        self.assertEqual(a.dtype, torch.float32)
        self.assertTrue(a.is_contiguous())
        torch.testing.assert_close(a[:, 0, 0], torch.tensor([-1., 128 / 127.5 - 1, 1.]))
        torch.testing.assert_close(b[:, 0, 0], torch.tensor([1., -1., 128 / 127.5 - 1]))
        torch.testing.assert_close((a + 1) / 2, torch.tensor([0., 128 / 255, 1.])[:, None, None].expand_as(a))

    def test_transform_preserves_spatial_pairing(self):
        clean = Image.new('RGB', (8, 8))
        clean.putpixel((2, 5), (255, 255, 255))
        a, b = paired_model_transform(clean, clean.copy())
        torch.testing.assert_close(a, b)
        self.assertTrue((a[:, 5, 2] == 1).all())
        self.assertEqual(int((a == 1).sum()), 3)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        Image.new('RGB', (8, 8)).save(self.root / 'clean.png')
        Image.new('RGB', (8, 8)).save(self.root / 'cloudy.png')
        self.manifest = self.root / 'pairs.json'
        self.record = dict(id='a', clean='clean.png', cloudy='cloudy.png')

    def write_manifest(self, records):
        self.manifest.write_text(json.dumps(records), encoding='utf-8')

    def test_pair_loading(self):
        self.write_manifest([self.record])
        dataset = PairedRGBDataset(self.manifest, pair_transform=test_transform, image_size=8)
        self.assertEqual(dataset[0]['id'], 'a')
        self.assertEqual(dataset[0]['clean'].shape, (3, 8, 8))

    def test_duplicate_ids_rejected(self):
        self.write_manifest([self.record, self.record])
        with self.assertRaises(ValueError):
            PairedRGBDataset(self.manifest, pair_transform=test_transform, image_size=8)

    def test_missing_pair_rejected(self):
        self.record['cloudy'] = 'missing.png'
        self.write_manifest([self.record])
        with self.assertRaises(FileNotFoundError):
            PairedRGBDataset(self.manifest, pair_transform=test_transform, image_size=8)

    def test_no_silent_resizing(self):
        self.write_manifest([self.record])
        dataset = PairedRGBDataset(self.manifest, pair_transform=test_transform)
        with self.assertRaises(ValueError):
            dataset[0]


if __name__ == '__main__':
    unittest.main()
