import unittest

from dizertatie.configs import DATASET_CONFIG_TESTS
from dizertatie.dataset import load
from dizertatie.model.robart_seq2seq import RoBartSeq2Seq, RoBartSeq2SeqConfig
from dizertatie.model.robert_classification import (
    RoBertClassifier,
    RoBertClassifierConfig,
)
from dizertatie.training.metrics import ClassificationMetrics, SummarizationMetrics
from tests.testcase import TestCaseWithData


class ModelTestCase(TestCaseWithData):
    def test_robert_model_for_ro_sent(self):
        dataset = load(DATASET_CONFIG_TESTS, "RoSent")

        model_config = RoBertClassifierConfig(
            num_labels=dataset.features["target"].num_classes
        )
        model = RoBertClassifier(model_config)
        self.assertIsNotNone(model)

        batch = dataset.select(range(3))
        self.assertListEqual(batch["id"], [2874, 26814, 3128])

        tokenized_batch = self.__tokenize_batch(batch, model)
        outputs = model.model(**tokenized_batch)
        self.assertTrue(hasattr(outputs, "logits"))
        self.assertTrue(hasattr(outputs, "loss"))
        self.assertTrue(outputs.loss > 0)

        metrics = ClassificationMetrics()
        logits, labels = (
            outputs.logits.detach().numpy(),
            tokenized_batch["labels"].detach().numpy(),
        )
        result = metrics.compute((logits, labels))
        self.assertIsInstance(result, dict)

    def test_load_robart_model_for_ro_text_summarization(self):
        dataset = load(DATASET_CONFIG_TESTS, "RoTextSummarization")

        model_config = RoBartSeq2SeqConfig()
        model = RoBartSeq2Seq(model_config)
        self.assertIsNotNone(model)

        batch = dataset.select(range(3))
        self.assertListEqual(batch["id"], [58679, 63956, 14193])

        tokenized_batch = self.__tokenize_batch(batch, model)
        outputs = model.model.generate(
            **tokenized_batch, max_new_tokens=16, do_sample=False
        )

        metrics = SummarizationMetrics(model.tokenizer)
        predictions, labels = (
            outputs.detach().numpy(),
            tokenized_batch["labels"].detach().numpy(),
        )
        result = metrics.compute((predictions, labels))
        self.assertIsInstance(result, dict)

    @staticmethod
    def __tokenize_batch(batch, model):
        tokenized_batch = batch.map(
            lambda x: model.tokenize(x, 16),
            remove_columns=list(
                filter(
                    lambda x: x in batch.column_names,
                    ["id", "text_ro", "target_ro", "stratify"],
                )
            ),
            batched=True,
        ).rename_column("target", "labels")
        # BUG: .to_dict() is messing up with .with_format('torch')
        return tokenized_batch.with_format("torch")[:]


if __name__ == "__main__":
    unittest.main()
