import unittest

from datasets import Dataset

from dizertatie.configs import DATASET_CONFIG_TESTS, TRAINING_CONFIG_TESTS
from dizertatie.dataset import load
from dizertatie.model.bart_seq2seq import BartSeq2Seq, BartSeq2SeqConfig
from dizertatie.model.base import BaseModel
from dizertatie.model.bert_classification import BertClassifier, BertClassifierConfig
from dizertatie.training.args import make_training_args
from dizertatie.training.metrics import ClassificationMetrics, SummarizationMetrics
from dizertatie.training.train import train_and_evaluate


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
        model = BartSeq2Seq(BartSeq2SeqConfig())

        tokenized_dataset = self.__tokenize_dataset(dataset, model)

        results = train_and_evaluate(
            model,
            make_training_args(TRAINING_CONFIG_TESTS, seq2seq=True),
            SummarizationMetrics(model.tokenizer),
            tokenized_dataset.select(range(0, 5)),
            tokenized_dataset.select(range(5, 10)),
        )

        self.assertTrue("eval_loss" in results)

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
