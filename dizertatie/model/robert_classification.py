import dataclasses

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dizertatie.model.base import BaseModel


@dataclasses.dataclass
class RoBertClassifierConfig:
    num_labels: int
    base_model: str = "dumitrescustefan/bert-base-romanian-cased-v1"


class RoBertClassifier(BaseModel):
    def __init__(self, config: RoBertClassifierConfig):
        super().__init__()
        self._tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model,
            ignore_mismatched_sizes=True,
            problem_type="single_label_classification",
            num_labels=config.num_labels,
        )

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model
