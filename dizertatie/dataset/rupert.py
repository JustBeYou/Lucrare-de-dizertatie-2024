import pathlib

import pandas
from datasets import Dataset
from pandas import DataFrame

from dizertatie.dataset.dataset import Config, DatasetDownloader


class Rupert(DatasetDownloader):
    n_top_authors = 25

    @staticmethod
    def _download_and_prepare(config: Config) -> Dataset:
        raw_path = pathlib.Path.joinpath(config.path, "raw_rupert.json")
        dataframe: DataFrame = pandas.read_json(raw_path)

        dataframe = dataframe.drop(dataframe[dataframe["tip"] == "strain"].index)

        to_keep = list(
            dataframe.groupby("autor").size().nlargest(Rupert.n_top_authors).index
        )
        dataframe = dataframe.drop(dataframe[~dataframe["autor"].isin(to_keep)].index)

        dataframe["text_ro"] = dataframe[["titlu", "text"]].agg("\n".join, axis=1)

        dataset = Dataset.from_pandas(dataframe[["text_ro", "autor"]])
        dataset = dataset.rename_column("autor", "target")
        dataset = dataset.remove_columns(["__index_level_0__"])
        return dataset
