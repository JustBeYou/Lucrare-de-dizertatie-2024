import abc


class BaseModel(abc.ABC):
    def __init__(self):
        pass

    @property
    @abc.abstractmethod
    def model(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def tokenizer(self):
        raise NotImplementedError

    def tokenize(self, examples: dict, max_tokens: int) -> dict:
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
