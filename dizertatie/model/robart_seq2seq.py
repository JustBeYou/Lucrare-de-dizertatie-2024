import dataclasses

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from dizertatie.model.base import BaseModel


@dataclasses.dataclass
class RoBartSeq2SeqConfig:
    base_model: str = "Iulian277/ro-bart-512"


class RoBartSeq2Seq(BaseModel):
    def __init__(self, config: RoBartSeq2SeqConfig):
        super().__init__()
        self._tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model)

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model