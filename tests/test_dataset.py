import pathlib
import unittest

from datasets import Dataset

from dizertatie.configs import DATASET_CONFIG
from dizertatie.dataset.dataset import load
from dizertatie.dataset.ro_sent import RoSent
from dizertatie.dataset.ro_text_summarization import RoTextSummarization
from dizertatie.dataset.rupert import Rupert


class DatasetTestCase(unittest.TestCase):
    def test_download_ro_sent(self):
        RoSent.download(DATASET_CONFIG)
        self.assertTrue(
            pathlib.Path.joinpath(DATASET_CONFIG.path, "download", "ro_sent").exists()
        )

    def test_download_ro_text_summarization(self):
        RoTextSummarization.download(DATASET_CONFIG)
        self.assertTrue(
            pathlib.Path.joinpath(
                DATASET_CONFIG.path, "download", "ro_text_summarization"
            ).exists()
        )

    def test_download_rupert(self):
        Rupert.download(DATASET_CONFIG)
        self.assertTrue(
            pathlib.Path.joinpath(DATASET_CONFIG.path, "download", "rupert").exists()
        )

    def test_load(self):
        dataset = load(DATASET_CONFIG, "download/rupert")
        self.assertTrue(isinstance(dataset, Dataset))
        self.assertTrue("text_ro" in dataset.column_names)


if __name__ == "__main__":
    unittest.main()
