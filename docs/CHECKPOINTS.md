# Using checkpoints

Train a teacher and student using [the training guide](REPRODUCING.md), then
use the resulting checkpoints for evaluation or inference. Keep each training
checkpoint with its configuration and dataset manifests for continuation.

For matched student evaluation, supply the teacher checkpoint used during
distillation and its SHA256. For deployment, export the student EMA weights:

```bash
python -m scripts.export_inference --checkpoint checkpoints/student_full.pt --output checkpoints/student_inference.pt --trust-checkpoint
python -m cloud_removal.predict --checkpoint checkpoints/student_inference.pt --input example_cloudy.png --output outputs/example_clear.png
```

Full training checkpoints use PyTorch serialization and can contain executable
pickle payloads. Only load trusted files after checking their hashes. The export
utility requires explicit `--trust-checkpoint`, reads the source without changing
it, and emits a self-contained EMA inference package with its own SHA256. Inference uses
`weights_only=True`. Full optimizer/RNG states are needed for exact continuation,
but not for inference. Check third-party terms before distributing any weights.
