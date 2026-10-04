# Third-party components

## k-diffusion / HDiT

- Upstream: https://github.com/crowsonkb/k-diffusion
- Source revision: `4601bf085320592473f681a62808ed873d17fad5`
- License: MIT, reproduced in the bundled source directory.
- Directory: `third_party/k-diffusion-4601bf085320592473f681a62808ed873d17fad5`
- Selected Python package and packaging files are copied from the research
  workspace. Build products, pretrained weights, and unrelated scripts are omitted.
- These files are a workspace snapshot based on the recorded upstream revision.
  Per-file hashes are in `docs/source_snapshot.json`.

## Runtime dependencies

PyTorch, torchvision, NATTEN, LPIPS, and the dependencies of k-diffusion retain
their respective licenses. Installation fetches these projects independently;
this repository does not grant rights to their code or weights.

## Dataset

CUHK-CR images are not redistributed. Consult the original authors' distribution
and terms, linked in `configs/dataset_source.json`. Only split identifiers and
pixel checksums are included. No third-party baseline model weights are bundled.
