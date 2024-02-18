import dataclasses
from pathlib import Path
from typing import Optional

from transformers import Seq2SeqTrainingArguments, TrainingArguments


@dataclasses.dataclass
class TrainingConfig:
    wandb_experiment: Optional[str]

    batch_size: int
    epochs: int
    output_dir: Path


__DEFAULT_ARGS = {
    "learning_rate": 2e-5,
    "weight_decay": 1e-2,
    "evaluation_strategy": "epoch",
    "load_best_model_at_end": False,
}


def make_training_args(
    config: TrainingConfig, seq2seq: bool = False
) -> TrainingArguments:
    common_args = dict(__DEFAULT_ARGS)
    common_args["output_dir"] = config.output_dir
    common_args["per_device_train_batch_size"] = config.batch_size
    common_args["per_device_eval_batch_size"] = config.batch_size
    common_args["num_train_epochs"] = config.epochs
    common_args["report_to"] = (
        "wandb" if config.wandb_experiment is not None else "none"
    )
    common_args["run_name"] = config.wandb_experiment

    if seq2seq:
        return Seq2SeqTrainingArguments(predict_with_generate=True, **common_args)

    return TrainingArguments(**common_args)
