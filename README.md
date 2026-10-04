# Single-step Cloud Removal with HDiT Consistency Distillation

Research implementation of an optical-conditioned EDM teacher, a single-step
consistency student, and a parameter-matched Direct HDiT regression control.

This repository provides the model implementation and tools for training,
evaluation, and inference in your own environment.

## Status

- Source: teacher training, pure student distillation, inference, evaluation,
  Direct regression, and core tests.
- Data: original CUHK-CR1 (534/134) and CUHK-CR2 (448/111) train/test identifiers;
  images must be obtained separately.
- Weights: **not included and no public download has been published yet**.
  See [checkpoint usage](docs/CHECKPOINTS.md) for using your trained models.
- License: the owner has not selected a license for original project code.
  See [licensing status](LICENSE_STATUS.md) and third-party notices.
- Run commands from this repository root. Linux/WSL with CUDA is the training
  target; neighborhood attention uses NATTEN.

## Installation

```bash
conda env create -f environment.yml
conda activate cloud-removal-release
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_*.py'
```

PyTorch 2.2.2 / CUDA 12.1 and NATTEN 0.17.3 are the reference stack. The NATTEN
wheel is Python/platform-specific; the shown environment targets Linux Python
3.10. Select compatible packages for your GPU, CUDA version, and operating system.
LPIPS may download its AlexNet backbone on first evaluation. Set `TORCH_HOME`
to a suitable cache location. Never load untrusted PyTorch checkpoints.

## Data

Obtain the dataset from its original distributor and follow [data/README.md](data/README.md).
Prepare 512 x 512 RGB inputs with the original pair order and image dimensions.
Validate before training:

```bash
python -m scripts.verify_data --subset CUHK-CR1
```

## Training and Evaluation

See [REPRODUCING.md](docs/REPRODUCING.md) for training commands, checkpoint
continuation settings, and the evaluation protocol.

The student uses one network evaluation at inference. Its 18-level grid is used
for training adjacent-noise consistency pairs. The teacher evaluation uses 18
Heun schedule points and 35 network evaluations.

## Cloudy-only Inference

With a trusted full checkpoint obtained separately, export a self-contained
EMA inference package (no optimizer, server paths, or random-generator states):

```bash
python -m scripts.export_inference --checkpoint checkpoints/student_full.pt --output checkpoints/student_inference.pt --trust-checkpoint
python -m cloud_removal.predict --checkpoint checkpoints/student_inference.pt --input example_cloudy.png --output outputs/example_clear.png
```

The prediction command does not require a clean target or a teacher checkpoint.
It validates RGB/size, uses BF16 on CUDA, and refuses to overwrite an output.

## Layout

```text
cloud_removal/   model, losses, sampling, training and evaluation
configs/        explicit architecture, data and experiment configurations
scripts/        data verification and inference export
tests/          CPU synthetic unit tests and release checks
data/           split manifests and pixel checksums, no images
docs/           training, evaluation and checkpoint usage
third_party/    selected HDiT dependency source with original license
```

Research-server launchers, credentials, private paths, data archives, full
training logs, binaries, and backup tools are intentionally excluded.
