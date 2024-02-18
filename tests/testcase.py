import unittest

from dizertatie.dataset import DATASETS
from tests.configs import DATASET_CONFIG_TESTS


class TestCaseWithData(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        for dataset_class in DATASETS:
            dataset_class.download(DATASET_CONFIG_TESTS)
