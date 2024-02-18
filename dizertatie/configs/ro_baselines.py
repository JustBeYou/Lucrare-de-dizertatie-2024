### Experiment Set 1: Romanian baselines ###
from typing import List

from dizertatie.configs.common import DATA_PATH, PROJECT_SEED, TRAIN_DATA_PATH
from dizertatie.dataset.dataset import DatasetConfig
from dizertatie.experiment.run import ExperimentConfig
from dizertatie.experiment.wandb import WandbConfig
from dizertatie.model.bart_seq2seq import BartSeq2Seq, BartSeq2SeqConfig
from dizertatie.model.bert_classification import BertClassifier, BertClassifierConfig
from dizertatie.training.args import TrainingConfig
from dizertatie.training.metrics import ClassificationMetrics, SummarizationMetrics
from dizertatie.training.split import CrossValidationConfig


def get_ro_baselines_config() -> List[ExperimentConfig]:
    __report_config = WandbConfig(project=f"Dizertatie - 2 - 5-fold CV Results")
    __cv_config = CrossValidationConfig(shuffle_seed=PROJECT_SEED, k_folds=5)

    return [
        ### RO_TEXT_SUMMARIZATION ###
        ExperimentConfig(
            dataset_name="RoTextSummarization",
            dataset_config=DatasetConfig(
                subsample_size=8_000,
                shuffle_seed=PROJECT_SEED,
                path=DATA_PATH,
            ),
            model_class=BartSeq2Seq,
            model_config=BartSeq2SeqConfig(
                base_model="Iulian277/ro-bart-1024",
                max_tokens=990,  # ~ 95% texts have fewer than this tokens
            ),
            train_config=TrainingConfig(
                batch_size=8,
                epochs=10,
                output_dir=TRAIN_DATA_PATH,
                generation_max_length=110,  # ~95% summarizes have fewer than this tokens
            ),
            metrics_class=SummarizationMetrics,
            cross_validation_config=__cv_config,
            report_config=__report_config,
        ),
        ### RO_SENT ###
        ExperimentConfig(
            dataset_name="RoSent",
            dataset_config=DatasetConfig(
                subsample_size=4_000,
                shuffle_seed=PROJECT_SEED,
                path=DATA_PATH,
            ),
            model_class=BertClassifier,
            model_config=BertClassifierConfig(
                base_model="dumitrescustefan/bert-base-romanian-cased-v1",
                max_tokens=208,  # ~ 95% of texts have fewer tokens
                num_labels=2,
            ),
            train_config=TrainingConfig(
                batch_size=32, epochs=10, output_dir=TRAIN_DATA_PATH
            ),
            metrics_class=ClassificationMetrics,
            cross_validation_config=__cv_config,
            report_config=__report_config,
        ),
        ### RUPERT ###
        ExperimentConfig(
            dataset_name="Rupert",
            dataset_config=DatasetConfig(
                subsample_size=5_000,
                shuffle_seed=PROJECT_SEED,
                path=DATA_PATH,
            ),
            model_class=BertClassifier,
            model_config=BertClassifierConfig(
                base_model="dumitrescustefan/bert-base-romanian-cased-v1",
                max_tokens=512,  # ~ 90% of poems have fewer tokens
                num_labels=25,
            ),
            train_config=TrainingConfig(
                batch_size=16, epochs=10, output_dir=TRAIN_DATA_PATH
            ),
            metrics_class=ClassificationMetrics,
            cross_validation_config=__cv_config,
            report_config=__report_config,
        ),
    ]
