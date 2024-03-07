import abc
import dataclasses
import functools
from typing import Optional


@dataclasses.dataclass
class BaseModelConfig(abc.ABC):
    max_tokens: int = 16


class BaseModel(abc.ABC):
    def __init__(self, config: BaseModelConfig):
        self.base_config = config

    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def model(self):
        raise NotImplementedError

    @functools.cached_property
    @abc.abstractmethod
    def tokenizer(self):
        raise NotImplementedError

    @abc.abstractmethod
    def to(self, device):
        raise NotImplementedError

    def tokenize(self, examples: dict, max_tokens: Optional[int] = None) -> dict:
        if max_tokens is None:
            max_tokens = self.base_config.max_tokens

        args = {
            "truncation": True,
            "padding": "max_length",
            "max_length": max_tokens,
        }

        result = self.tokenizer(examples["text_ro"], **args)

        if "target_ro" in examples:
            tokenized_target = self.tokenizer(examples["target_ro"], **args)
            result["target"] = tokenized_target["input_ids"]

        return result
