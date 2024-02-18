import dataclasses
import gc
from typing import Type

import torch
import wandb
from datasets import Dataset

from dizertatie.dataset.dataset import DatasetConfig, load
from dizertatie.experiment.wandb import WandbConfig, wandb_init
from dizertatie.model.base import BaseModel, BaseModelConfig
from dizertatie.training.args import TrainingConfig, make_training_args
from dizertatie.training.metrics import Metrics
from dizertatie.training.split import CrossValidationConfig, split_k_fold
from dizertatie.training.train import train_and_evaluate


@dataclasses.dataclass
class ExperimentConfig:
    dataset_name: str
    dataset_config: DatasetConfig

    model_config: BaseModelConfig
    model_class: Type[BaseModel]
    train_config: TrainingConfig
    metrics_class: Type[Metrics]

    cross_validation_config: CrossValidationConfig
    report_config: WandbConfig


def run_experiment(config: ExperimentConfig):
    dataset = load(config.dataset_config, config.dataset_name)
    tokenizer_model = config.model_class(config.model_config)

    dataset = dataset.map(tokenizer_model.tokenize, batched=True)
    dataset = dataset.with_format("torch")

    run_name = f"{config.dataset_name}_{tokenizer_model.name}"
    print(f"### Running experiment: {run_name} ###")
    print(f"## W&B project: {config.report_config.project}")
    print(f"## GPU available: {torch.cuda.is_available()}")
    print(f"## Dataset name: {config.dataset_name}")
    print(f"## Model name: {config.model_class.__name__}")
    print(f"## Metrics name: {config.metrics_class.__name__}")
    print(f"## Cross validation with K: {config.cross_validation_config.k_folds}")

    try:
        folds = split_k_fold(dataset, config.cross_validation_config)
        seq2seq = "seq2seq" in config.model_class.__name__.lower()

        wandb_init(config.report_config)

        for i, (train_set, test_set) in enumerate(folds):
            print(f"# Fold: {i + 1}/{config.cross_validation_config.k_folds}")
            trainable_model = config.model_class(config.model_config)
            results = train_and_evaluate(
                trainable_model,
                make_training_args(
                    config.train_config,
                    run_name=run_name,
                    report_to="wandb",
                    seq2seq=seq2seq,
                ),
                config.metrics_class(tokenizer_model),
                __prepare_columns(train_set),
                __prepare_columns(test_set),
            )
            print(f"# Results: {results}")

            __clear_memory_cache()
    finally:
        print("### Finished experiment ###")
        wandb.finish()
        wandb._cleanup_media_tmp_dir()


def __prepare_columns(dataset: Dataset) -> Dataset:
    return dataset.remove_columns(
        list(
            filter(
                lambda x: x in dataset.column_names,
                ["id", "text_ro", "target_ro", "stratify"],
            )
        )
    ).rename_column("target", "labels")


def __clear_memory_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
