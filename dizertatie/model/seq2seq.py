import dataclasses
import functools

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from dizertatie.model.base import BaseModel, BaseModelConfig


@dataclasses.dataclass
class ModelSeq2SeqConfig(BaseModelConfig):
    base_model: str = "Iulian277/ro-bart-512"


class ModelSeq2Seq(BaseModel):
    def __init__(self, config: ModelSeq2SeqConfig):
        super().__init__(config)
        self.config = config

    @property
    def name(self):
        return f"Model-{self.config.base_model.replace('/', '-')}"

    @functools.cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.config.base_model)

    @functools.cached_property
    def model(self):
        return AutoModelForSeq2SeqLM.from_pretrained(self.config.base_model)