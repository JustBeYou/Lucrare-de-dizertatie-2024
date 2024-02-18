import abc
from typing import Dict, Optional, Tuple

import evaluate
import numpy
import torch

from dizertatie.model.base import BaseModel


class Metrics(abc.ABC):
    def __init__(self, model: Optional[BaseModel] = None):
        self.model = model

    @abc.abstractmethod
    def compute(self, eval_pred: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        raise NotImplementedError


class ClassificationMetrics(Metrics):
    def __init__(self, model: Optional[BaseModel] = None):
        super().__init__(model)
        metrics = ["f1", "precision", "recall", "accuracy"]
        modes = ["macro", "weighted"]

        self.metrics = [evaluate.load(metric) for metric in metrics]
        self.modes = modes

    def compute(self, eval_pred: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        logits, labels = eval_pred
        predictions = numpy.argmax(logits, axis=-1)
        results = {}

        for metric in self.metrics:
            if metric.name == "accuracy":
                result = metric.compute(predictions=predictions, references=labels)
                results[metric.name] = result[metric.name]
            else:
                for mode in self.modes:
                    result = metric.compute(
                        predictions=predictions, references=labels, average=mode
                    )
                    results[f"{metric.name}_{mode}"] = result[metric.name]

        return results


class SummarizationMetrics(Metrics):
    def __init__(self, model: BaseModel):
        super().__init__(model)
        if model is None:
            raise ValueError("model is required")

        self.tokenizer = model.tokenizer
        self.rouge = evaluate.load("rouge")

    def compute(self, eval_pred: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        predictions, labels = eval_pred
        predictions = numpy.where(
            predictions != -100, predictions, self.tokenizer.pad_token_id
        )
        decoded_predictions = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        labels = numpy.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = self.rouge.compute(
            predictions=decoded_predictions, references=decoded_labels, use_stemmer=True
        )

        prediction_lens = [
            numpy.count_nonzero(pred != self.tokenizer.pad_token_id)
            for pred in predictions
        ]
        result["gen_len"] = numpy.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}

        return result
