import os
import unittest

from dizertatie.experiment.run import run_experiment
from tests.configs import RUN_EXPERIMENT_CONFIG_TESTS
from tests.testcase import TestCaseWithData


class ExperimentTestCase(TestCaseWithData):
    def test_run_experiment_classification(self):
        self.__skip_if_no_wandb()
        run_experiment(RUN_EXPERIMENT_CONFIG_TESTS["classification"])
        self.assertTrue(True)

    def test_run_experiment_summarization(self):
        self.__skip_if_no_wandb()
        run_experiment(RUN_EXPERIMENT_CONFIG_TESTS["summarization"])
        self.assertTrue(True)

    def __skip_if_no_wandb(self):
        if "WANDB_API_KEY" not in os.environ:
            self.skipTest("Wandb API key must be provided.")


if __name__ == "__main__":
    unittest.main()
