import dataclasses
from pathlib import Path
from typing import Optional

from transformers import TrainingArguments


@dataclasses.dataclass
class TrainingConfig:
    wandb_experiment: Optional[str]

    batch_size: int
    epochs: int
    output_dir: Path


class CustomTrainingArguments(TrainingArguments):
    def __init__(self, config: TrainingConfig):
        super().__init__(
            output_dir=str(config.output_dir),
            learning_rate=2e-5,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            num_train_epochs=config.epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            load_best_model_at_end=False,
            report_to="wandb" if config.wandb_experiment is not None else "none",
            run_name=config.wandb_experiment,
        )
