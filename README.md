# Single-step Cloud Removal with HDiT Consistency Distillation

Research implementation of an optical-conditioned EDM teacher, a single-step
consistency student, and a parameter-matched Direct HDiT regression control.

This release packages the working research implementation, not a claim that
every manuscript result has been reproduced. Archived local measurements and
their provenance are reported separately in [results](results/README.md).
No new training is started by installation or tests.

## Status

- Source: teacher training, pure student distillation, inference, evaluation,
  Direct regression, and core tests.
- Data: original CUHK-CR1 (534/134) and CUHK-CR2 (448/111) train/test identifiers;
  images must be obtained separately. No held-out split is created.
- Weights: **not included and no public download has been published yet**.
  See [checkpoint inventory](docs/CHECKPOINTS.md).
- License: the owner has not selected a license for original project code.
  See [licensing status](LICENSE_STATUS.md) and third-party notices.
- Run commands from this repository root. Linux/WSL with CUDA is the training
  target; the original attention path uses NATTEN with no silent replacement.

## Installation

```bash
conda env create -f environment.yml
conda activate cloud-removal-release
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_*.py'
```

PyTorch 2.2.2 / CUDA 12.1 and NATTEN 0.17.3 are the reference stack. The NATTEN
wheel is Python/platform-specific; the shown environment targets Linux Python
3.10. Some transitive dependencies are not fully locked; see validation notes.
LPIPS may download its AlexNet backbone on first evaluation. Set `TORCH_HOME`
to a suitable cache location. Never load untrusted PyTorch checkpoints.

## Data

Obtain the dataset from its original distributor and follow [data/README.md](data/README.md).
No resizing, RGB conversion, augmentation, or duplicate removal is performed
implicitly. Inputs must be 512 x 512 RGB. Validate before training:

```bash
python -m scripts.verify_data --subset CUHK-CR1
```

## Training and Evaluation

See [REPRODUCING.md](docs/REPRODUCING.md) for runnable commands, the distinction
between new training and historical continuation, and the matched metric protocol.

The student uses one network evaluation at inference. Its 18-level grid is used
for training adjacent-noise consistency pairs. The teacher evaluation uses 18
Heun schedule points and 35 network evaluations. No 2/3-step student algorithm
is asserted by this release's one-step evaluation entry point.

## Cloudy-only Inference

With a trusted full checkpoint obtained separately, export a self-contained
EMA inference package (no optimizer, server paths, or random-generator states):

```bash
python -m scripts.export_inference --checkpoint checkpoints/student_full.pt --output checkpoints/student_inference.pt --trust-checkpoint
python -m cloud_removal.predict --checkpoint checkpoints/student_inference.pt --input example_cloudy.png --output outputs/example_clear.png
```

The prediction command does not require a clean target or a teacher checkpoint.
It validates RGB/size, uses BF16 on CUDA, and refuses to overwrite an output.
The export and prediction commands are release utilities, not new experiments.

## Layout

```text
cloud_removal/   model, losses, sampling, training and evaluation
configs/        explicit architecture, data and experiment configurations
scripts/        data verification and inference export
tests/          CPU synthetic unit tests and release checks
data/           split manifests and pixel checksums, no images
docs/           reproduction, checkpoint and validation notes
results/        selected archived local measurements
third_party/    selected HDiT dependency source with original license
```

Research-server launchers, credentials, private paths, data archives, full
training logs, binaries, and backup tools are intentionally excluded.
