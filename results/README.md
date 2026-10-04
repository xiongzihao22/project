# Matched evaluation results

Results from the local Direct100 comparison evaluated on 2026-10-03.

CUHK-CR1 original 134 test pairs; 512 x 512 RGB; RTX 4090; BF16; batch 1;
seed 700013; three warmup batches/calls as implemented in the evaluator.
Time is synchronized prediction only. All models have 41,621,203 parameters.

| Model | PSNR | SSIM | RMSE | LPIPS | NFE | s/image |
|---|---:|---:|---:|---:|---:|---:|
| Direct20 | 24.2218 | .6742 | .06282 | .35430 | 1 | .02327 |
| Direct100 | 25.7027 | .7378 | .05392 | .20167 | 1 | .02323 |
| Teacher800 | 25.3638 | .6991 | .05894 | .17095 | 35 | .76331 |
| Student20 | 24.4519 | .6878 | .06286 | .19202 | 1 | .02399 |

Direct100 is better on PSNR, SSIM and RMSE here, while teacher/student have lower
LPIPS. Direct20 and student20 have equal update counts. Direct training starts
from random initialization, while the student uses teacher pretraining and EMA
initialization. The Direct100 training budget is 100 epochs.

`direct_teacher_student_cr1.json` contains full-precision metrics, selected
protocol fields, original result-file hashes and checkpoint hashes. Private
machine paths and raw operational logs are excluded. The result-file hashes
identify the original measurement records. NFE counts network forward evaluations;
s/image reports measured prediction latency.
