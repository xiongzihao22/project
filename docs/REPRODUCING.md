# Training and evaluation

All commands below run from the repository root. Output directories must be new.
Training uses CUDA and can incur substantial compute cost; no command is run
automatically. Installation and release validation do not retrain models.

## New teacher training

```bash
python -m cloud_removal.train_teacher --subset CUHK-CR1 --config configs/teacher_training.json --output-dir outputs/teacher --target-epoch 300
```

This is the included experimental 300-epoch baseline recipe with AdamW 1e-4,
not a one-command reproduction of the historical 800-epoch teacher. Resume only
with a matching configuration using `--resume outputs/teacher/latest.pt`.

`configs/teacher800_cosine_archived.json` is the exact saved late-stage
configuration. It is provided for inspection and compatible continuation, **not
as a from-scratch 800-epoch recipe**. The archived lineage used earlier constant
learning-rate stages, a low-LR fork at 700, cosine decay from global step 96480
to 107200, and recovery from a saved checkpoint after interruption. Starting
from random weights with the final configuration does not recreate this history.
Earlier source checkpoints and transitions are not bundled in this release.

## Student from an epoch-800 teacher

```bash
python -m cloud_removal.train_student --subset CUHK-CR1 --config configs/student800_pure20.json --teacher-checkpoint checkpoints/teacher800_full.pt --output-dir outputs/student20 --target-epoch 20 --milestone-epochs 1
```

The supplied config requires an actual epoch-800 teacher. The student initializes
from the teacher EMA backbone, creates a fresh optimizer, and uses consistency
loss without added endpoint or L1 reconstruction supervision. AdamW 1e-5, batch
4, EMA .999, gradient clipping .01, seed 13, Euler adjacent-state updates, and
an 18-level Karras grid are explicit. The teacher stays frozen.

## Direct control

```bash
python -m cloud_removal.train_direct_hdit --subset CUHK-CR1 --output outputs/direct100 --epochs 100 --save-epochs 20 --learning-rate 1e-4 --loss l1 --seed 13
```

Direct uses random initialization, concat(zero, cloudy), constant time 1, and
L1 supervision, without a teacher. Its 20-epoch checkpoint matches student
update count, not teacher-plus-student training cost or initialization.

## Matched evaluation

```bash
python -m cloud_removal.evaluate_evidence --method teacher --checkpoint checkpoints/teacher800_full.pt --subset CUHK-CR1 --steps 18 --output outputs/eval_teacher --seed 700013 --batch-size 1 --warmup 3
python -m cloud_removal.evaluate_evidence --method student --checkpoint outputs/student20/latest.pt --teacher-checkpoint checkpoints/teacher800_full.pt --expected-teacher-sha256 TEACHER_SHA256 --subset CUHK-CR1 --steps 1 --output outputs/eval_student --seed 700013 --batch-size 1 --warmup 3
python -m cloud_removal.evaluate_evidence --method direct --checkpoint outputs/direct100/latest.pt --subset CUHK-CR1 --steps 1 --output outputs/eval_direct --seed 700013 --batch-size 1 --warmup 3
```

Replace `TEACHER_SHA256` with the independently checked hash of the exact teacher
used to train that student. Archived reference hashes are in CHECKPOINTS.md.
The formal student evaluator intentionally requires the source teacher to verify
provenance; the deployed student predictor does not require the teacher.

The evaluator uses the full fixed test set, BF16, synchronized prediction-only
wall time, PSNR, SSIM, RMSE on [0,1] scale, and LPIPS-Alex v0.1. Loading, metrics,
and warmup are outside prediction timing. NFE is checked with a forward hook.
Do not compare these BF16 timings to independently measured FP32 table entries
without noting the different protocol. LPIPS dependencies/weights must be
available separately. No DiffCR factory or external-baseline benchmark package
is bundled; the historical optional evaluator branch is not a supplied baseline.

Single-seed results on a repeatedly evaluated test set are not independent
validation. Low error on one metric does not establish universal superiority.
The code supports CR2 manifests, but this release's archived quality table is
CR1 only; it does not imply that the corresponding CR2 weights were trained.
