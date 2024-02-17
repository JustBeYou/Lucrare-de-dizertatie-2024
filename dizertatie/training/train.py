from typing import Dict

from datasets import Dataset
from transformers import Trainer

from dizertatie.model.base import BaseModel
from dizertatie.training.args import CustomTrainingArguments
from dizertatie.training.metrics import Metrics


def train_and_evaluate(
    model: BaseModel,
    args: CustomTrainingArguments,
    metrics: Metrics,
    train_set: Dataset,
    test_set: Dataset,
) -> Dict[str, float]:
    trainer = Trainer(
        model=model.model,
        args=args,
        train_dataset=train_set,
        eval_dataset=test_set,
        tokenizer=model.tokenizer,
        compute_metrics=metrics.compute,
    )

    trainer.train()

    return trainer.evaluate()
