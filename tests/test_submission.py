import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.submission import save_submission


class TestSubmission(unittest.TestCase):

    def test_creates_missing_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:

            output_path = (
                Path(temp_dir)
                / "artifacts"
                / "submissions"
                / "submission.csv"
            )

            submission = pd.DataFrame({
                "Id": [1461, 1462],
                "SalePrice": [150000.0, 200000.0]
            })

            save_submission(submission, output_path)

            self.assertTrue(output_path.exists())

            loaded = pd.read_csv(output_path)

            self.assertEqual(loaded.shape, (2, 2))
            self.assertEqual(
                loaded.columns.tolist(),
                ["Id", "SalePrice"]
            )
            self.assertTrue(
                loaded["SalePrice"].notna().all()
            )


if __name__ == "__main__":
    unittest.main()