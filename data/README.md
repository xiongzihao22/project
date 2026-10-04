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

Portable manifest bytes differ from historical absolute-path manifests; their
SHA256 values must not be represented as the historical manifest hashes. Strict
evaluation of an archived Direct checkpoint will intentionally reject a changed
training-manifest hash. Do not bypass that check or edit the archived checkpoint.
New training against these portable manifests records the new hashes. Cloudy-only
prediction through an exported inference package does not require manifests.
