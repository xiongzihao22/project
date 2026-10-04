import unittest

from cloud_removal.train_direct_hdit import budget_allows_next_epoch


class DirectBudgetTests(unittest.TestCase):
    def test_unlimited(self):
        self.assertTrue(budget_allows_next_epoch(1000, None, 100))

    def test_reserve_last_epoch_duration(self):
        self.assertTrue(budget_allows_next_epoch(100, 201, 100))
        self.assertFalse(budget_allows_next_epoch(100, 200, 100))

    def test_expired(self):
        self.assertFalse(budget_allows_next_epoch(300, 200, 100))


if __name__ == '__main__':
    unittest.main()
