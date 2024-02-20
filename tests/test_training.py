import unittest

from datasets import Dataset

from dizertatie.dataset import load
from dizertatie.model.base import BaseModel
from dizertatie.model.bert_classification import BertClassifier, BertClassifierConfig
from dizertatie.model.seq2seq import ModelSeq2Seq, ModelSeq2SeqConfig
from dizertatie.training.args import make_training_args
from dizertatie.training.metrics import ClassificationMetrics, SummarizationMetrics
from dizertatie.training.split import split_k_fold
from dizertatie.training.train import train_and_evaluate
from tests.configs import (
    CROSS_VALIDATION_CONFIG_TESTS,
    DATASET_CONFIG_TESTS,
    TRAINING_CONFIG_TESTS,
)


class TrainingTestCase(unittest.TestCase):
    def test_training_classification(self):
        dataset = load(DATASET_CONFIG_TESTS, "RoSent")
        model = BertClassifier(
            BertClassifierConfig(num_labels=dataset.features["target"].num_classes)
        )

        tokenized_dataset = self.__tokenize_dataset(dataset, model)

        results = train_and_evaluate(
            model,
            make_training_args(TRAINING_CONFIG_TESTS),
            ClassificationMetrics(),
            tokenized_dataset.select(range(0, 5)),
            tokenized_dataset.select(range(5, 10)),
        )

        self.assertTrue("eval_loss" in results)

    def test_training_summarization(self):
        dataset = load(DATASET_CONFIG_TESTS, "RoTextSummarization")
        model = ModelSeq2Seq(ModelSeq2SeqConfig())

        tokenized_dataset = self.__tokenize_dataset(dataset, model)

        results = train_and_evaluate(
            model,
            make_training_args(TRAINING_CONFIG_TESTS, seq2seq=True),
            SummarizationMetrics(model),
            tokenized_dataset.select(range(0, 5)),
            tokenized_dataset.select(range(5, 10)),
        )

        self.assertTrue("eval_loss" in results)

    def test_split_k_fold(self):
        dataset = load(DATASET_CONFIG_TESTS, "Rupert")
        splits_generator = split_k_fold(dataset, CROSS_VALIDATION_CONFIG_TESTS)

        train_1, test_1 = next(splits_generator)
        train_2, test_2 = next(splits_generator)

        print(train_1["id"][0], train_2["id"][0], test_1["id"][0], test_2["id"][0])

        self.assertEqual(train_1["id"][0], 4257)
        self.assertEqual(train_2["id"][0], 4071)
        self.assertEqual(test_1["id"][0], 4071)
        self.assertEqual(test_2["id"][0], 4257)

        train_test_intersect = set(train_1["id"]).intersection(set(test_1["id"]))
        self.assertTrue(len(train_test_intersect) == 0)

    @staticmethod
    def __tokenize_dataset(dataset: Dataset, model: BaseModel) -> Dataset:
        return (
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


if __name__ == "__main__":
    unittest.main()
