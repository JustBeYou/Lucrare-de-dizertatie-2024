import unittest

from dizertatie.configs import DATASET_CONFIG_TESTS
from dizertatie.dataset import load
from dizertatie.model.robert_classification import (
    RoBertClassifier,
    RoBertClassifierConfig,
)
from tests.testcase import TestCaseWithData


class ModelTestCase(TestCaseWithData):
    def test_robert_model_for_ro_sent(self):
        dataset = load(DATASET_CONFIG_TESTS, "RoSent")

        model_config = RoBertClassifierConfig(
            num_labels=dataset.features["target"].num_classes
        )
        model = RoBertClassifier(model_config)
        self.assertIsNotNone(model)

        dataset = dataset.rename_column("target", "labels")
        batch = dataset.select(range(3))
        self.assertListEqual(batch["id"], [2874, 26814, 3128])

        tokenized_batch = batch.map(
            lambda x: model.tokenize(x, 16),
            remove_columns=["id", "text_ro"],
            batched=True,
        )
        # BUG: .to_dict() is messing up with .with_format('torch')
        tokenized_batch = tokenized_batch.with_format("torch")[:]
        outputs = model.model(**tokenized_batch)
        self.assertTrue(hasattr(outputs, "logits"))
        self.assertTrue(hasattr(outputs, "loss"))
        self.assertTrue(outputs.loss > 0)


if __name__ == "__main__":
    unittest.main()
0
