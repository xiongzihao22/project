import unittest
import torch
from torch import nn
from cloud_removal.hdit_adapter import ConditionedHDiT


class RecordingInterface(nn.Module):
    def forward(self, image, sigma):
        self.image = image
        self.sigma = sigma
        return image[:, :3] + image[:, 3:]


class AdapterTests(unittest.TestCase):
    def test_channel_order_and_raw_sigma(self):
        fixture = RecordingInterface()
        adapter = ConditionedHDiT(fixture, input_transform=lambda x, s, y: (x, y))
        x = torch.zeros(2, 3, 8, 8, requires_grad=True)
        y = torch.ones_like(x)
        result = adapter(x, 2., y)
        self.assertEqual(fixture.image.shape, (2, 6, 8, 8))
        self.assertTrue(torch.equal(fixture.image[:, :3], x))
        self.assertTrue(torch.equal(fixture.image[:, 3:], y))
        self.assertTrue(torch.equal(fixture.sigma, torch.full((2,), 2.)))
        result.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_zero_sigma_rejected(self):
        adapter = ConditionedHDiT(RecordingInterface(), input_transform=lambda x, s, y: (x, y))
        with self.assertRaises(ValueError):
            adapter(torch.zeros(1, 3, 8, 8), 0., torch.zeros(1, 3, 8, 8))
