# Release validation (2026-10-04)

- 96 CPU synthetic/core tests passed with Python 3.10 and PyTorch 2.2.2 in an
  existing WSL environment. This is not a fresh-machine installation test.
- Exported the actual archived epoch-20 student full checkpoint, verified its
  source SHA256, and reloaded the EMA-only package with `weights_only=True`.
- One actual 512 x 512 RGB cloudy image was reconstructed on a local RTX 3060
  Laptop GPU using the bundled HDiT source, CUDA and BF16. Output size and finite
  values were checked by the prediction entry point. This is a smoke check,
  not a latency benchmark or a new quality evaluation.
- The exported checkpoint and smoke image are ignored local artifacts and are
  not part of the GitHub source upload.
- No server training, full-dataset reevaluation, paper edits or original Git
  index changes were performed during release preparation.

The reference environment specifies PyTorch, CUDA, NATTEN, LPIPS, NumPy and
einops versions, but not every transitive k-diffusion dependency is locked.
The current local WSL test environment lacks LPIPS; full four-metric evaluation
was not rerun there. The LPIPS recipe is supplied for the target environment.
Archived quality measurements are historical results, not release-test outputs.

Model weights are not yet publicly downloadable. Exact reconstruction of the
historical 800-epoch trajectory additionally needs the earlier checkpoints and
transition history, which are not provided as a complete from-scratch recipe.
