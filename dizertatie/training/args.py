import dataclasses
from pathlib import Path
from typing import Optional

from transformers import Seq2SeqTrainingArguments, TrainingArguments


@dataclasses.dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    output_dir: Path
    generation_max_length: Optional[int] = None


__DEFAULT_ARGS = {
    "learning_rate": 2e-5,
    "weight_decay": 1e-2,
    "evaluation_strategy": "epoch",
    "load_best_model_at_end": False,
}


def make_training_args(
    config: TrainingConfig,
    run_name: Optional[str] = None,
    report_to: str = "none",
    seq2seq: bool = False,
) -> TrainingArguments:
    common_args = dict(__DEFAULT_ARGS)
    common_args["output_dir"] = config.output_dir
    common_args["per_device_train_batch_size"] = config.batch_size
    common_args["per_device_eval_batch_size"] = config.batch_size
    common_args["num_train_epochs"] = config.epochs
    common_args["report_to"] = report_to
    common_args["run_name"] = run_name

    if seq2seq:
        if config.generation_max_length is None:
            raise ValueError("generation_max_length is required")

        return Seq2SeqTrainingArguments(
            predict_with_generate=True,
            generation_max_length=config.generation_max_length,
            **common_args
        )

    return TrainingArguments(**common_args)
