import pathlib

from dizertatie.dataset.dataset import Config

DATASET_CONFIG = Config(path=pathlib.Path(__file__).parent.parent.joinpath("data"))
