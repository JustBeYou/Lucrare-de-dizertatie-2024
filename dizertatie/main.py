import pathlib

from dizertatie.configs.common import PROJECT_SEED
from dizertatie.dataset import RoSent, RoTextSummarization, Rupert
from dizertatie.dataset.dataset import DatasetConfig
from dizertatie.experiment.run import run_experiment


def main():
    config = DatasetConfig(
        subsample_size=0,
        shuffle_seed=PROJECT_SEED,
        path=pathlib.Path(__file__).parent.parent.joinpath("data"),
        force_overwrite=False,
    )

    RoSent.download(config)
    RoTextSummarization.download(config)
    Rupert.download(config)

    configs = []

    for config in configs:
        run_experiment(config)


if __name__ == "__main__":
    main()
