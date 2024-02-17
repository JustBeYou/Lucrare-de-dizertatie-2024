from typing import Dict, List, Tuple

import evaluate
import numpy
import torch


class ClassificationMetrics:
    def __init__(self):
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


class SummarizationMetrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.rouge = evaluate.load("rouge")

    def compute(self, eval_pred: Tuple[torch.Tensor, torch]):
        predictions, labels = eval_pred
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
