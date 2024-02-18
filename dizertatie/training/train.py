from typing import Dict, Union

from datasets import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
)

from dizertatie.model.base import BaseModel
from dizertatie.training.metrics import Metrics


def train_and_evaluate(
    model: BaseModel,
    args: Union[TrainingArguments, Seq2SeqTrainingArguments],
    metrics: Metrics,
    train_set: Dataset,
    test_set: Dataset,
) -> Dict[str, float]:
    trainer_class = Trainer

    if isinstance(args, Seq2SeqTrainingArguments):
        trainer_class = Seq2SeqTrainer

    trainer = trainer_class(
        model=model.model,
        args=args,
        train_dataset=train_set,
        eval_dataset=test_set,
        tokenizer=model.tokenizer,
        compute_metrics=metrics.compute,
    )

    trainer.train()

    return trainer.evaluate()
