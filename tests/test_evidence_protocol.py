import copy
import unittest
from cloud_removal.evidence_protocol import compare, expected_nfe, validate_result


class EvidenceTests(unittest.TestCase):
    def test_heun_counts_calls_not_steps(self):
        self.assertEqual(expected_nfe('teacher', 18), 35)
        self.assertEqual(expected_nfe('teacher', 2), 3)

    def test_single_step(self):
        for method in ('student', 'direct', 'diffcr'):
            self.assertEqual(expected_nfe(method, 1), 1)

    def test_reject_bad_steps(self):
        for method, steps in [('teacher', 1), ('student', 2), ('direct', 20), ('diffcr', 0)]:
            with self.assertRaises(ValueError):
                expected_nfe(method, steps)

    def result(self):
        metrics = dict(psnr=25, ssim=.7, lpips=.2, rmse=.05)
        return dict(schema_version=2, status='completed', scope='full_test',
                    method='student', steps=1, weights='ema', checkpoint_sha256='student',
                    provenance={'teacher': {'checkpoint_sha256': 'teacher', 'weights': 'ema'}},
                    manifest_sha256='a', metric_protocol='b', lpips_protocol='c',
                    train_manifest_sha256='train', data_protocol_sha256='data',
                    image_size=512, batch_size=1, precision='bf16', device='gpu', seed=13,
                    torch_version='2', cuda_version='12', input_content_sha256='inputs',
                    initial_noise_sha256='noise', metric_code_sha256='metrics',
                    lpips_state_sha256='lpips', timing_protocol='predict_only',
                    sample_ids=['a', 'b'], test_split_size=2, metrics=metrics,
                    per_image=[{'id': identity, **metrics} for identity in ('a', 'b')],
                    batches=[{'samples': 1, 'actual_nfe': 1, 'inference_seconds': 5.0}] * 2,
                    inference_seconds=10.0, nfe=1)

    def test_comparison(self):
        reference = self.result()
        candidate = copy.deepcopy(reference)
        candidate['inference_seconds'] = 2
        candidate['batches'] = [{'samples': 1, 'actual_nfe': 1, 'inference_seconds': 1.0}] * 2
        candidate['metrics']['psnr'] -= .2
        for row in candidate['per_image']:
            row['psnr'] -= .2
        report = compare(reference, candidate)
        self.assertEqual(report['reference_over_candidate_latency'], 5)
        self.assertAlmostEqual(report['candidate_minus_reference']['psnr'], -.2)

    def test_protocol_mismatch(self):
        for key in ('manifest_sha256', 'batch_size', 'precision', 'sample_ids', 'seed'):
            candidate = self.result()
            candidate[key] = 'different'
            with self.assertRaises(ValueError):
                compare(self.result(), candidate)

    def test_nonfinite(self):
        candidate = self.result()
        candidate['metrics']['psnr'] = float('nan')
        with self.assertRaises(ValueError):
            compare(self.result(), candidate)

    def test_reject_smoke_and_incomplete(self):
        for key, value in (('scope', 'smoke_only'), ('status', 'running'),
                           ('sample_ids', ['a', 'a']), ('test_split_size', 134)):
            result = self.result()
            result[key] = value
            with self.assertRaises(ValueError):
                validate_result(result)

    def test_reject_actual_nfe_mismatch(self):
        result = self.result()
        result['batches'][0]['actual_nfe'] = 2
        with self.assertRaises(ValueError):
            validate_result(result)

    def test_reject_mean_and_timing_inconsistency(self):
        for key, value in (('inference_seconds', float('nan')),
                           ('inference_seconds', 8), ('metrics', dict(psnr=24, ssim=.7, lpips=.2, rmse=.05))):
            result = self.result()
            result[key] = value
            with self.assertRaises(ValueError):
                validate_result(result)

    def test_teacher_must_match_student_source(self):
        teacher = self.result()
        teacher.update(method='teacher', steps=2, nfe=3, checkpoint_sha256='teacher')
        teacher['batches'] = [{'samples': 1, 'actual_nfe': 3, 'inference_seconds': 5.0}] * 2
        self.assertEqual(compare(teacher, self.result())['reference_over_candidate_nfe'], 3)
        teacher['checkpoint_sha256'] = 'different-teacher'
        with self.assertRaises(ValueError):
            compare(teacher, self.result())
        report = compare(teacher, self.result(), additional_teacher_baseline=True)
        self.assertEqual(report['comparison_scope'],
                         'additional_teacher_baseline_not_student_distillation_source')
        teacher['seed'] = 99
        with self.assertRaises(ValueError):
            compare(teacher, self.result(), additional_teacher_baseline=True)


if __name__ == '__main__':
    unittest.main()
