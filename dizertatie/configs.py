import pathlib

from dizertatie.dataset.dataset import Config

PROJECT_SEED = 12022000

DATASET_CONFIG_TESTS = Config(
    subsample_size=100,
    shuffle_seed=PROJECT_SEED,
    path=pathlib.Path(__file__).parent.parent.joinpath("data"),
)
