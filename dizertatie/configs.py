import pathlib

from dizertatie.dataset.dataset import DatasetConfig
from dizertatie.training.args import TrainingConfig

PROJECT_SEED = 12022000

DATASET_CONFIG_TESTS = DatasetConfig(
    subsample_size=100,
    shuffle_seed=PROJECT_SEED,
    path=pathlib.Path(__file__).parent.parent.joinpath("data"),
)

TRAINING_CONFIG_TESTS = TrainingConfig(
    epochs=1,
    batch_size=1,
    output_dir=pathlib.Path(__file__).parent.parent.joinpath("data", "training"),
    wandb_experiment=None,
)
