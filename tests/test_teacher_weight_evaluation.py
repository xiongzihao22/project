import unittest

import torch

from cloud_removal.evaluate_teacher_weights import (
    metric_deltas,
    state_distance,
    validate_checkpoint,
)


class TeacherWeightEvaluationTests(unittest.TestCase):
    def test_state_distance(self):
        online = {"weight": torch.tensor([3.0, 4.0]), "count": torch.tensor(1)}
        ema = {"weight": torch.tensor([0.0, 0.0]), "count": torch.tensor(1)}
        result = state_distance(online, ema)
        self.assertEqual(result["floating_elements"], 2)
        self.assertAlmostEqual(result["online_l2"], 5.0)
        self.assertAlmostEqual(result["difference_l2"], 5.0)
        self.assertAlmostEqual(result["relative_l2"], 1.0)
        self.assertAlmostEqual(result["maximum_absolute_difference"], 4.0)

    def test_nonfloating_state_difference_rejected(self):
        with self.assertRaises(ValueError):
            state_distance({"count": torch.tensor(1)}, {"count": torch.tensor(2)})

    def test_metric_delta_signs_are_ema_minus_online(self):
        online = [{"sigma": 1.0, "samples": 2, "mse": 0.2, "psnr": 10.0, "ssim": 0.3}]
        ema = [{"sigma": 1.0, "samples": 2, "mse": 0.1, "psnr": 12.0, "ssim": 0.4}]
        result = metric_deltas(online, ema)[0]
        self.assertAlmostEqual(result["ema_minus_online_mse"], -0.1)
        self.assertAlmostEqual(result["ema_minus_online_psnr"], 2.0)
        self.assertAlmostEqual(result["ema_minus_online_ssim"], 0.1)

    def test_checkpoint_identity(self):
        config = {"a": 1}
        checkpoint = {
            "kind": "full_training_state",
            "subset": "CUHK-CR1",
            "epoch": 50,
            "global_step": 1,
            "config": config,
            "model": {},
            "ema_model": {},
        }
        validate_checkpoint(checkpoint, "CUHK-CR1", config, 50)
        with self.assertRaises(ValueError):
            validate_checkpoint(checkpoint, "CUHK-CR2", config, 50)


if __name__ == "__main__":
    unittest.main()
