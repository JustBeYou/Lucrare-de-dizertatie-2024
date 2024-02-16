from typing import Tuple

from datasets import Dataset, load_dataset

from dizertatie.dataset.dataset import Config, DatasetDownloader


class RoSent(DatasetDownloader):
    @staticmethod
    def _download_and_prepare(_config: Config) -> Tuple[str, Dataset]:
        dataset = load_dataset("ro_sent", split="all")
        dataset = dataset.remove_columns(["original_id", "id"])
        dataset = dataset.rename_column("sentence", "text_ro")
        dataset = dataset.rename_column("label", "target")
        return "ro_sent", dataset
