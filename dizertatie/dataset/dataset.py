import abc
import dataclasses
import pathlib
from typing import Optional

from datasets import ClassLabel, Dataset, load_from_disk


@dataclasses.dataclass
class Config:
    subsample_size: Optional[int]
    shuffle_seed: int

    path: pathlib.Path
    force_overwrite: bool = False


def load(config: Config, name: str) -> Dataset:
    path = pathlib.Path.joinpath(config.path, name)
    if not path.exists():
        path = pathlib.Path.joinpath(config.path, "download", name)

    dataset = load_from_disk(str(path))
    if config.subsample_size is None:
        return dataset

    return __stratified_subsample(config, dataset)


def __stratified_subsample(config: Config, dataset: Dataset) -> Dataset:
    stratify_column = "stratify" if "stratify" in dataset.features else "target"

    return dataset.train_test_split(
        train_size=config.subsample_size,
        shuffle=True,
        seed=config.shuffle_seed,
        stratify_by_column=stratify_column,
    )["train"]


class DatasetDownloader(abc.ABC):
    @classmethod
    def download(cls, config: Config):
        download_path = pathlib.Path.joinpath(config.path, "download")
        if not download_path.exists():
            download_path.mkdir()

        download_path = download_path.joinpath(cls.__name__)
        if download_path.exists() and not config.force_overwrite:
            return

        dataset = cls._download_and_prepare(config)
        cls.__check_dataset(dataset)

        # Romanian datasets must conform to Romanian characters
        dataset = dataset.map(cls.__fix_diacritics_batched, batched=True)

        # ID column for predictability
        id_column = list(range(len(dataset)))
        dataset = dataset.add_column("id", id_column)

        # Prepare a column for stratified sampling
        for column in ["stratify", "target"]:
            if column in dataset.features and not isinstance(
                dataset.features[column], ClassLabel
            ):
                dataset = dataset.class_encode_column(column)

        dataset.save_to_disk(download_path)

    @staticmethod
    @abc.abstractmethod
    def _download_and_prepare(config: Config) -> Dataset:
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
