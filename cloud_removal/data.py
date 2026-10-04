"""Explicit paired RGB manifest; no automatic splitting or normalization.

JSON: [{"id": "sample", "clean": "clear.png", "cloudy": "cloud.png"}].
Relative paths are resolved against the manifest parent. A transform receives
both images together and must return a pair of tensors in the chosen range.
"""
import json
import os
import platform
from pathlib import Path, PureWindowsPath

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset


def paired_model_transform(clean, cloudy):
    """Convert paired RGB images to float32 CHW [-1, 1], without augmentation."""
    def convert(image):
        if image.mode != 'RGB':
            raise ValueError('Expected RGB image')
        pixels = np.array(image, dtype=np.float32, copy=True)
        return torch.from_numpy(pixels).permute(2, 0, 1).contiguous().div(255).mul(2).sub(1)
    return convert(clean), convert(cloudy)


def resolve_image_path(root, value):
    """Support existing drive-letter manifests on WSL without rewriting them."""
    windows = PureWindowsPath(value)
    is_wsl = (
        bool(os.environ.get('WSL_DISTRO_NAME') or os.environ.get('WSL_INTEROP'))
        or 'microsoft' in platform.release().lower()
    )
    if is_wsl and windows.is_absolute() and len(windows.drive) == 2:
        return Path('/mnt') / windows.drive[0].lower() / Path(*windows.parts[1:])
    return root / value


class PairedRGBDataset(Dataset):
    def __init__(self, manifest, *, pair_transform, image_size=512):
        manifest = Path(manifest)
        self.samples = json.loads(manifest.read_text(encoding='utf-8'))
        if not isinstance(self.samples, list) or not self.samples:
            raise ValueError('Manifest must be a nonempty list')
        self.root = manifest.resolve().parent
        self.pair_transform = pair_transform
        self.image_size = image_size
        ids = set()
        for sample in self.samples:
            if not isinstance(sample, dict) or not {'id', 'clean', 'cloudy'} <= sample.keys():
                raise ValueError('Each pair needs id, clean and cloudy fields')
            if sample['id'] in ids:
                raise ValueError('Duplicate sample id')
            ids.add(sample['id'])
            for field in ('clean', 'cloudy'):
                path = resolve_image_path(self.root, sample[field])
                if not path.is_file():
                    raise FileNotFoundError(path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        images = []
        for field in ('clean', 'cloudy'):
            with Image.open(resolve_image_path(self.root, sample[field])) as image:
                if image.mode != 'RGB':
                    raise ValueError('Expected RGB; band conversion must be explicit')
                if image.size != (self.image_size, self.image_size):
                    raise ValueError('Unexpected source resolution; no implicit resizing')
                images.append(image.copy())
        clean, cloudy = self.pair_transform(*images)
        if clean.shape != cloudy.shape or tuple(clean.shape) != (3, self.image_size, self.image_size):
            raise ValueError('Transform must preserve paired CHW RGB image dimensions')
        return {'id': sample['id'], 'clean': clean, 'cloudy': cloudy}
