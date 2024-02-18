import dataclasses
import functools

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dizertatie.model.base import BaseModel, BaseModelConfig


@dataclasses.dataclass
class BertClassifierConfig(BaseModelConfig):
    num_labels: int = 2
    base_model: str = "dumitrescustefan/bert-base-romanian-cased-v1"


class BertClassifier(BaseModel):
    def __init__(self, config: BertClassifierConfig):
        super().__init__(config)
        self.config = config

    @property
    def name(self):
        return f"Bert-{self.config.base_model.replace('/', '-')}"

    @functools.cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.config.base_model)

    @functools.cached_property
    def model(self):
        return AutoModelForSequenceClassification.from_pretrained(
            self.config.base_model,
            ignore_mismatched_sizes=True,
            problem_type="single_label_classification",
            num_labels=self.config.num_labels,
        )
