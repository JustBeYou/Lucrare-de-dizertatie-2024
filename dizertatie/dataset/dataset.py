import abc
import dataclasses
import pathlib

from datasets import Dataset, load_from_disk


@dataclasses.dataclass
class Config:
    path: pathlib.Path
    force_overwrite: bool = False


def load(config: Config, name: str) -> Dataset:
    path = pathlib.Path.joinpath(config.path, name)
    if not path.exists():
        path = pathlib.Path.joinpath(config.path, "download", name)

    return load_from_disk(str(path))


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

        dataset = dataset.map(cls.__fix_diacritics_batched, batched=True)
        id_column = list(range(len(dataset)))
        dataset = dataset.add_column("id", id_column)
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
