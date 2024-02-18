import dataclasses
from typing import Iterator, Tuple

import numpy
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold


@dataclasses.dataclass
class CrossValidationConfig:
    k_folds: int
    shuffle_seed: int


def split_k_fold(
    dataset: Dataset, config: CrossValidationConfig
) -> Iterator[Tuple[Dataset, Dataset]]:
    dataset = dataset.shuffle(seed=config.shuffle_seed)
    k_fold = StratifiedKFold(
        n_splits=config.k_folds, shuffle=True, random_state=config.shuffle_seed
    )
    stratify_column = "stratify" if "stratify" in dataset.features else "target"
    splits = k_fold.split(numpy.zeros(dataset.num_rows), dataset[stratify_column])

    for train_idx, test_idx in splits:
        train_set = dataset.select(train_idx)
        test_set = dataset.select(test_idx)

        yield train_set, test_set
