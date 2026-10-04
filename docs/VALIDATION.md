# Release validation (2026-10-04)

- 96 CPU synthetic/core tests passed with Python 3.10 and PyTorch 2.2.2 in an
  existing WSL environment.
- Exported the actual archived epoch-20 student full checkpoint, verified its
  source SHA256, and reloaded the EMA-only package with `weights_only=True`.
- One actual 512 x 512 RGB cloudy image was reconstructed on a local RTX 3060
  Laptop GPU using the bundled HDiT source, CUDA and BF16. Output size and finite
  values were checked by the prediction entry point.
- The exported checkpoint and smoke image are ignored local artifacts and are
  not part of the GitHub source upload.

The reference environment pins PyTorch, CUDA, NATTEN, LPIPS, NumPy and einops;
transitive dependencies follow their package constraints. Release checks cover
unit tests, checkpoint export, and single-image inference. Four-metric evaluation
requires the target environment's LPIPS installation and weights.

Dataset-level quality results are recorded in [results](../results/README.md).
Checkpoint availability and training settings are documented in
[CHECKPOINTS.md](CHECKPOINTS.md) and [REPRODUCING.md](REPRODUCING.md).
