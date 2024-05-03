import abc
import dataclasses
import pathlib
from typing import Optional, List

import pandas
from datasets import ClassLabel, Dataset, load_from_disk


@dataclasses.dataclass
class DatasetConfig:
    subsample_size: Optional[int]
    shuffle_seed: int

    path: pathlib.Path
    force_overwrite: bool = False


def load(config: DatasetConfig, name: str) -> Dataset:
    path = pathlib.Path.joinpath(config.path, name)
    if not path.exists():
        path = pathlib.Path.joinpath(config.path, "download", name)

    dataset = load_from_disk(str(path))
    if config.subsample_size is None:
        return dataset

    return __stratified_subsample(config, dataset)


@dataclasses.dataclass
class TranslationConfig:
    path: pathlib.Path
    translator: str


def translate_dataset(dataset: Dataset, config: TranslationConfig) -> Dataset:
    translations = pandas.read_json(str(config.path))
    assert translations["id"].to_list() == dataset["id"]

    # Translate input text
    dataset = dataset.remove_columns(["text_ro"])
    dataset = dataset.add_column("text_ro", translations[f"text_en_{config.translator}"].to_list())

    # Translate output text if necessary
    if "target_ro" in dataset.column_names:
        dataset = dataset.remove_columns(["target_ro", "target"])
        target_en = translations[f"target_en_{config.translator}"].to_list()
        dataset = dataset.add_column("target", target_en)
        dataset = dataset.add_column("target_ro", target_en)

    return dataset

def load_translations(dataset: Dataset, path: str, translators: List[str]) -> Dataset:
    translations = pandas.read_json(str(path))
    print(path, translations.columns)
    assert translations["id"].to_list() == dataset["id"]

    for translator in translators:
        dataset = dataset.add_column(f"text_en_{translator}", translations[f"text_en_{translator}"].to_list())
        if f'target_en_{translator}' in translations.columns:
            dataset = dataset.add_column(f'target_en_{translator}', translations[f"target_en_{translator}"].to_list())

    return dataset

def __stratified_subsample(config: DatasetConfig, dataset: Dataset) -> Dataset:
    stratify_column = "stratify" if "stratify" in dataset.features else "target"

    return dataset.train_test_split(
        train_size=config.subsample_size,
        shuffle=True,
        seed=config.shuffle_seed,
        stratify_by_column=stratify_column,
    )["train"]


class DatasetDownloader(abc.ABC):
    @classmethod
    def download(cls, config: DatasetConfig):
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
    def _download_and_prepare(config: DatasetConfig) -> Dataset:
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
