import dataclasses
import os
from typing import Optional

import wandb


@dataclasses.dataclass
class WandbConfig:
    project: str
    api_key: Optional[str] = None


def wandb_init(config: WandbConfig):
    api_key = (
        config.api_key if config.api_key is not None else os.environ["WANDB_API_KEY"]
    )
    wandb.login(key=api_key)
    os.environ["WANDB_PROJECT"] = config.project
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_RUN_GROUP"] = "parallel"
