from typing import Tuple

from datasets import Dataset, load_dataset

from dizertatie.dataset.dataset import Config, DatasetDownloader


class RoTextSummarization(DatasetDownloader):
    @staticmethod
    def _download_and_prepare(_config: Config) -> Tuple[str, Dataset]:
        dataset: Dataset = load_dataset(
            "readerbench/ro-text-summarization", split="all"
        )
        dataset = dataset.map(
            RoTextSummarization.__concat_title_to_content_batched, batched=True
        )
        dataset = dataset.rename_column("Content", "text_ro")
        dataset = dataset.rename_column("Summary", "target")
        dataset = dataset.rename_column("Category", "stratify")
        dataset = dataset.add_column("target_ro", dataset["target"])
        dataset = dataset.remove_columns(["Title", "Source", "href"])
        return "ro_text_summarization", dataset

    @staticmethod
    def __concat_title_to_content_batched(examples: dict) -> dict:
        examples["Content"] = [
            f"Titlu: {title} Conținut: {content}"
            for title, content in zip(examples["Title"], examples["Content"])
        ]
        return examples
