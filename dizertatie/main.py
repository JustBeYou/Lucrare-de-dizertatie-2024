import pathlib
import sys
from pprint import pprint

from dizertatie.configs.common import PROJECT_SEED
from dizertatie.configs.en_translation import get_en_translation_config
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

    configs = get_en_translation_config()

    translator  = sys.argv[-1]
    if translator in configs:
        for config in configs[translator]:
            run_experiment(config)
    else:
        pprint(configs)

if __name__ == "__main__":
    main()
