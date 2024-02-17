import dataclasses
from typing import Optional

from transformers import TrainingArguments


@dataclasses.dataclass
class Config:
    wandb_experiment: Optional[str]

    batch_size: int
    epochs: int
    output_dir: str


class CustomTrainingArguments(TrainingArguments):
    def __init__(self, config: Config):
        super().__init__(
            output_dir=Config.output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=Config.batch_size,
            per_device_eval_batch_size=Config.batch_size,
            num_train_epochs=Config.epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            load_best_model_at_end=False,
            report_to="wandb" if config.wandb_experiment is not None else "none",
            run_name=config.wandb_experiment,
        )
