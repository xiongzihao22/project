# Training and evaluation

All commands below run from the repository root. Output directories must be new.
Training requires a CUDA GPU.

## New teacher training

```bash
python -m cloud_removal.train_teacher --subset CUHK-CR1 --config configs/teacher_training.json --output-dir outputs/teacher --target-epoch 300
```

This command trains the 300-epoch baseline with AdamW at 1e-4. Resume with a
matching configuration using `--resume outputs/teacher/latest.pt`.

`configs/teacher800_cosine_archived.json` provides a late-stage continuation
configuration with cosine decay from global step 96480 to 107200. Use it with a
compatible checkpoint. For a new run, start with `teacher_training.json` and
set the training duration and learning-rate schedule for that run.

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
L1 supervision. Its 20-epoch checkpoint and the student have the same update
count. The student additionally uses teacher pretraining and EMA initialization.

## Matched evaluation

```bash
python -m cloud_removal.evaluate_evidence --method teacher --checkpoint checkpoints/teacher800_full.pt --subset CUHK-CR1 --steps 18 --output outputs/eval_teacher --seed 700013 --batch-size 1 --warmup 3
python -m cloud_removal.evaluate_evidence --method student --checkpoint outputs/student20/latest.pt --teacher-checkpoint checkpoints/teacher800_full.pt --expected-teacher-sha256 TEACHER_SHA256 --subset CUHK-CR1 --steps 1 --output outputs/eval_student --seed 700013 --batch-size 1 --warmup 3
python -m cloud_removal.evaluate_evidence --method direct --checkpoint outputs/direct100/latest.pt --subset CUHK-CR1 --steps 1 --output outputs/eval_direct --seed 700013 --batch-size 1 --warmup 3
```

Replace `TEACHER_SHA256` with the independently checked hash of the exact teacher
used to train that student. See [checkpoint usage](CHECKPOINTS.md).
The student evaluator checks the source teacher checkpoint hash. Cloudy-only
deployment uses the exported student package.

The evaluator uses the full fixed test set, BF16, synchronized prediction-only
wall time, PSNR, SSIM, RMSE on [0,1] scale, and LPIPS-Alex v0.1. Loading, metrics,
and warmup are outside prediction timing. NFE is checked with a forward hook.
Timing comparisons use matching precision and measurement settings. Install the
LPIPS dependencies and weights before evaluation. External-baseline benchmarks
require their own model implementations and weights.

Choose `CUHK-CR1` or `CUHK-CR2` to evaluate your checkpoints with the corresponding
dataset manifests.
