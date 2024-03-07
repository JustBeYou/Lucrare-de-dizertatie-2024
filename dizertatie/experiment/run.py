import dataclasses
import gc
import os
from typing import Optional, Type, List

import torch
import wandb
from datasets import Dataset

import accelerate
from dizertatie.dataset.dataset import DatasetConfig, TranslationConfig, load, translate_dataset
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

    translation_config: Optional[TranslationConfig] = None
    torch_device = None


def run_experiments_parallel(configs: List[ExperimentConfig]):
    print(f"#### Running parallel experiments ####")
    distributed_state = accelerate.PartialState()
    for config in configs:
        config.torch_device = distributed_state.device

    ids = list(range(len(configs)))
    with distributed_state.split_between_processes(ids) as split_ids:
        print(f"### Spawning process {distributed_state.process_index} with experiments {split_ids}")
        split_configs = [configs[i] for i in split_ids]

        for config in split_configs:
            run_experiment(config)

def run_experiment(config: ExperimentConfig):
    dataset = load(config.dataset_config, config.dataset_name)
    if config.translation_config is not None:
        dataset = translate_dataset(dataset, config.translation_config)

    tokenizer_model = config.model_class(config.model_config)

    dataset = dataset.map(tokenizer_model.tokenize, batched=True)
    dataset = dataset.with_format("torch")

    run_name = f"{config.dataset_name}_{tokenizer_model.name}"
    if config.translation_config is not None:
        run_name = f"{run_name}_EN_{config.translation_config.translator}"

    print(f"### Running experiment: {run_name} ###")
    print(f"## W&B project: {config.report_config.project}")
    print(f"## GPU available: {torch.cuda.is_available()}")
    if config.torch_device:
        print(f"## GPU Device: {config.torch_device}")
    print(f"## Dataset name: {config.dataset_name}")
    print(f"## Translator: {config.translation_config.translator}")
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
            if config.torch_device:
                trainable_model.to(config.torch_device)
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
                torch_device=config.torch_device
            )
            print(f"# Results: {results}")

            __clear_memory_cache()
            wandb.finish()
    finally:
        print("### Finished experiment ###")
        wandb.finish()

        # os.system("rm -rf wandb")


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
        print(f"# CUDA Memory before free:", torch.cuda.mem_get_info())
        torch.cuda.empty_cache()
        print(f"# CUDA Memory after free:", torch.cuda.mem_get_info())
