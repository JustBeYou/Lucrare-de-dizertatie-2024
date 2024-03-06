import pathlib
import unittest

from datasets import Dataset

from dizertatie.dataset import DATASETS
from dizertatie.dataset.dataset import DatasetConfig, add_translations, load
from tests.configs import DATASET_CONFIG_TESTS, PROJECT_SEED
from tests.testcase import TestCaseWithData


class DatasetTestCase(TestCaseWithData):
    def test_load(self):
        for dataset_class in DATASETS:
            dataset = load(DATASET_CONFIG_TESTS, dataset_class.__name__)
            self.assertTrue(isinstance(dataset, Dataset))
            self.assertTrue("text_ro" in dataset.column_names)

    def test_load_translations_subsample(self):
        data_path = pathlib.Path(__file__).parent.parent.joinpath("data")
        translation_path = data_path.joinpath("rupert_subsample_translations.json")
        dataset_translation_config = DatasetConfig(
            subsample_size=5_000,
            shuffle_seed=PROJECT_SEED,
            path=data_path,
        )

        dataset = load(dataset_translation_config, "Rupert")
        translated_dataset = add_translations(dataset, translation_path)
        self.assertTrue(isinstance(translated_dataset, Dataset))
        self.assertTrue(
            translated_dataset[1337]["text_en_mistral"]
            .strip()
            .startswith("Psalm VI You stir up")
        )


if __name__ == "__main__":
    unittest.main()
