import unittest

from dizertatie.configs import DATASET_CONFIG_TESTS, TRAINING_CONFIG_TESTS
from dizertatie.dataset import load
from dizertatie.model.robert_classification import (
    RoBertClassifier,
    RoBertClassifierConfig,
)
from dizertatie.training.args import CustomTrainingArguments
from dizertatie.training.metrics import ClassificationMetrics
from dizertatie.training.train import train_and_evaluate


class TrainingTestCase(unittest.TestCase):
    def test_training_classification(self):
        dataset = load(DATASET_CONFIG_TESTS, "RoSent")
        model = RoBertClassifier(
            RoBertClassifierConfig(num_labels=dataset.features["target"].num_classes)
        )

        tokenized_dataset = (
            dataset.map(
                lambda x: model.tokenize(x, 16),
                remove_columns=list(
                    filter(
                        lambda x: x in dataset.column_names,
                        ["id", "text_ro", "target_ro", "stratify"],
                    )
                ),
                batched=True,
            )
            .rename_column("target", "labels")
            .with_format("torch")
        )

        results = train_and_evaluate(
            model,
            CustomTrainingArguments(TRAINING_CONFIG_TESTS),
            ClassificationMetrics(),
            tokenized_dataset.select(range(0, 5)),
            tokenized_dataset.select(range(5, 10)),
        )

        self.assertTrue("eval_loss" in results)


if __name__ == "__main__":
    unittest.main()
