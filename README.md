# Single-step Cloud Removal with HDiT Consistency Distillation

HDiT-based cloud removal with an EDM teacher and a single-step consistency student.

## Environment

The supplied environment targets Linux or WSL with an NVIDIA GPU.

| Component | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.2.2 |
| torchvision | 0.17.2 |
| CUDA runtime | 12.1 |
| NATTEN | 0.17.3 |
| NumPy | 1.26.4 |
| LPIPS | 0.1.4 |
| einops | 0.8.2 |

Run from the repository root:

```bash
conda env create -f environment.yml
conda activate cloud-removal-release
python -m pip install -r requirements.txt
```

NATTEN provides neighborhood attention. Its wheel must match the Python,
PyTorch, and CUDA versions. Dependencies are specified in `environment.yml`
and `requirements.txt`. LPIPS downloads its AlexNet weights on first use;
set `TORCH_HOME` to choose the cache directory.

Run the unit tests:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Code Structure

```text
cloud_removal/   models, losses, sampling, training, evaluation and inference
configs/        model and training configurations
scripts/        data verification, inference export and source checks
tests/          unit tests
data/           dataset split manifests and pixel checksums
third_party/    k-diffusion / HDiT dependency
```

Dataset layout is described in [data/README.md](data/README.md).
Dependency attribution is provided in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
