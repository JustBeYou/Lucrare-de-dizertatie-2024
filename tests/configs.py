import pathlib

from dizertatie.dataset.dataset import DatasetConfig
from dizertatie.experiment.run import ExperimentConfig
from dizertatie.experiment.wandb import WandbConfig
from dizertatie.model.bart_seq2seq import BartSeq2Seq, BartSeq2SeqConfig
from dizertatie.model.bert_classification import BertClassifier, BertClassifierConfig
from dizertatie.training.args import TrainingConfig
from dizertatie.training.metrics import ClassificationMetrics, SummarizationMetrics
from dizertatie.training.split import CrossValidationConfig

PROJECT_SEED = 12022000

MAX_INPUT_TOKENS = 16
MAX_OUTPUT_TOKENS = 8

DATASET_CONFIG_TESTS = DatasetConfig(
    subsample_size=100,
    shuffle_seed=PROJECT_SEED,
    path=pathlib.Path(__file__).parent.parent.joinpath("data"),
)

TRAINING_CONFIG_TESTS = TrainingConfig(
    epochs=1,
    batch_size=10,
    output_dir=pathlib.Path(__file__).parent.parent.joinpath("data", "training"),
    generation_max_length=MAX_OUTPUT_TOKENS,
)

CROSS_VALIDATION_CONFIG_TESTS = CrossValidationConfig(
    shuffle_seed=PROJECT_SEED, k_folds=2
)

REPORT_CONFIG_TESTS = WandbConfig(project="Dizertatie CI Tests")

DATASET_SMALL_CONFIG_TESTS = DatasetConfig(
    subsample_size=20,
    shuffle_seed=PROJECT_SEED,
    path=pathlib.Path(__file__).parent.parent.joinpath("data"),
)

RUN_EXPERIMENT_CONFIG_TESTS = {
    "classification": ExperimentConfig(
        dataset_name="RoSent",
        dataset_config=DATASET_SMALL_CONFIG_TESTS,
        model_class=BertClassifier,
        model_config=BertClassifierConfig(max_tokens=MAX_INPUT_TOKENS, num_labels=2),
        train_config=TRAINING_CONFIG_TESTS,
        metrics_class=ClassificationMetrics,
        cross_validation_config=CROSS_VALIDATION_CONFIG_TESTS,
        report_config=REPORT_CONFIG_TESTS,
    ),
    "summarization": ExperimentConfig(
        dataset_name="RoTextSummarization",
        dataset_config=DATASET_SMALL_CONFIG_TESTS,
        model_class=BartSeq2Seq,
        model_config=BartSeq2SeqConfig(max_tokens=MAX_INPUT_TOKENS),
        train_config=TRAINING_CONFIG_TESTS,
        metrics_class=SummarizationMetrics,
        cross_validation_config=CROSS_VALIDATION_CONFIG_TESTS,
        report_config=REPORT_CONFIG_TESTS,
    ),
}
