# Dataset preparation

Source: https://github.com/littlebeen/DDPM-Enhancement-for-Cloud-Removal

Download metadata and the original archive hash are in
`configs/dataset_source.json`. Check redistribution and use conditions with the
dataset authors. Images are not included in this repository.

Extract or arrange the original files as:

```text
data/raw/CUHK-CR/CUHK-CR1/train/label/*.png
data/raw/CUHK-CR/CUHK-CR1/train/cloud/*.png
data/raw/CUHK-CR/CUHK-CR1/test/label/*.png
data/raw/CUHK-CR/CUHK-CR1/test/cloud/*.png
```

CUHK-CR2 uses the same layout. `manifests/` retains the archived pair identities
and order but uses paths relative to each manifest. Counts are CR1 534 train /
134 test and CR2 448 train / 111 test. Do not deduplicate or repartition.

`checksums/` records SHA256 of the decoded RGB pixel bytes, not compressed PNG
file bytes. `python -m scripts.verify_data --subset CUHK-CR1` checks all pairs,
image size/mode, and those pixel hashes. It never modifies images.

Portable manifests use relative paths and have their own SHA256 identifiers.
Direct checkpoint evaluation requires an exact match to the training-manifest
hash stored in the checkpoint. For archived checkpoints, use the corresponding
original manifests; new training records the portable manifest hashes.
Cloudy-only prediction uses an exported inference package without manifests.
