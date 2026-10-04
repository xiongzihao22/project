"""Verify original split identities and decoded RGB checksums, without writes."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def verify(root, subset):
    protocol = json.loads((root / 'configs/data_protocol.json').read_text())
    counts = {}
    for split in ('train', 'test'):
        manifest = root / protocol['subsets'][subset][split + '_manifest']
        rows = json.loads(manifest.read_text())
        checks = json.loads((root / f'data/checksums/{subset}_{split}.json').read_text())
        if len(rows) != protocol['subsets'][subset][split + '_count']:
            raise ValueError('Original split count changed')
        if [r['id'] for r in rows] != [r['id'] for r in checks]:
            raise ValueError('Pair identity/order mismatch')
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate pair ID')
        for row, checksum in zip(rows, checks):
            for role in ('clean', 'cloudy'):
                with Image.open(manifest.parent / row[role]) as image:
                    if image.mode != 'RGB' or image.size != (512, 512):
                        raise ValueError(f'Invalid image mode/size: {row["id"]}')
                    if hashlib.sha256(image.tobytes()).hexdigest() != checksum[role + '_pixel_sha256']:
                        raise ValueError(f'Pixel hash mismatch: {split}/{row["id"]}/{role}')
        counts[split] = len(rows)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subset', choices=('CUHK-CR1', 'CUHK-CR2'), required=True)
    args = parser.parse_args()
    print(json.dumps(verify(Path(__file__).resolve().parents[1], args.subset)))


if __name__ == '__main__':
    main()
