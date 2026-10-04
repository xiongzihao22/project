from pathlib import Path
import tempfile
import unittest

from cloud_removal.train_direct_hdit import checkpoint_epochs, retain_checkpoint


class DirectRetentionTests(unittest.TestCase):
    def test_optional_and_sorted(self):
        self.assertEqual(checkpoint_epochs([], 100), [])
        self.assertEqual(checkpoint_epochs([100, 20], 100), [20, 100])

    def test_invalid(self):
        for epochs in ([0], [101], [20, 20], [1.5], [True]):
            with self.assertRaises(ValueError):
                checkpoint_epochs(epochs, 100)

    def test_independent_copy_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / 'latest.pt', Path(tmp) / 'epoch20.pt'
            source.write_bytes(b'complete state')
            retain_checkpoint(source, target)
            source.write_bytes(b'next state')
            self.assertEqual(target.read_bytes(), b'complete state')
            with self.assertRaises(FileExistsError):
                retain_checkpoint(source, target)
            self.assertFalse(target.with_suffix('.pt.tmp').exists())


if __name__ == '__main__':
    unittest.main()
