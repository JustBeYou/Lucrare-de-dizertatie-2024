import unittest

from datasets import Dataset

from dizertatie.configs import DATASET_CONFIG
from dizertatie.dataset import DATASETS
from dizertatie.dataset.dataset import load
from tests.testcase import TestCaseWithData


class DatasetTestCase(TestCaseWithData):
    def test_load(self):
        for dataset_class in DATASETS:
            dataset = load(DATASET_CONFIG, dataset_class.__name__)
            self.assertTrue(isinstance(dataset, Dataset))
            self.assertTrue("text_ro" in dataset.column_names)


if __name__ == "__main__":
    unittest.main()
