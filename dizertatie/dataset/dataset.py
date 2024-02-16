import abc
import dataclasses
import os
import pathlib
from typing import Tuple

from datasets import Dataset, load_from_disk


@dataclasses.dataclass
class Config:
    path: pathlib.Path


def load(config: Config, name: str) -> Dataset:
    path = pathlib.Path.joinpath(config.path, name)
    return load_from_disk(str(path))


class DatasetDownloader(abc.ABC):
    @classmethod
    def download(cls, config: Config):
        name, dataset = cls._download_and_prepare(config)
        cls.__check_dataset(dataset)

        dataset = dataset.map(cls.__fix_diacritics_batched, batched=True)
        id_column = list(range(len(dataset)))
        dataset = dataset.add_column("id", id_column)

        download_path = pathlib.Path.joinpath(config.path, "download")
        if not download_path.exists():
            download_path.mkdir()

        path = download_path.joinpath(name)
        dataset.save_to_disk(path)

    @staticmethod
    @abc.abstractmethod
    def _download_and_prepare(config: Config) -> Tuple[str, Dataset]:
        raise NotImplementedError

    @staticmethod
    def __check_dataset(dataset: Dataset):
        assert "text_ro" in dataset.column_names
        assert "target" in dataset.column_names

    @staticmethod
    def __fix_diacritics_batched(examples: dict) -> dict:
        """
        According to https://huggingface.co/dumitrescustefan/bert-base-romanian-cased-v1
        """
        examples["text_ro"] = [
            text.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
            for text in examples["text_ro"]
        ]
        return examples
