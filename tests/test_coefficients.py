import math
import unittest
from cloud_removal.coefficients import consistency_coefficients


class CoefficientTests(unittest.TestCase):
    def test_boundary(self):
        self.assertEqual(consistency_coefficients(0.02), (1.0, 0.0))

    def test_equation(self):
        skip, out = consistency_coefficients(1.0)
        self.assertAlmostEqual(skip, 0.25 / (0.98**2 + 0.25))
        self.assertAlmostEqual(out, 0.49 / math.sqrt(1.25))

    def test_custom_boundary(self):
        self.assertEqual(consistency_coefficients(0.1, sigma_min=0.1), (1.0, 0.0))

    def test_invalid_noise(self):
        for sigma in (-1, 0, 0.01, float('nan'), float('inf')):
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                consistency_coefficients(sigma)


if __name__ == '__main__':
    unittest.main()
