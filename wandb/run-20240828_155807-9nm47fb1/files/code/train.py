import hydra
from omegaconf import DictConfig
import argparse
import sys
import dotenv

import pyrootutils
import pdb

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", ".project-root"],
    pythonpath=True,
    dotenv=True,
)

# # load environment variables from `.env` file if it exists
# # recursively searches for `.env` in all folders starting from work dir
# dotenv.load_dotenv(override=True)

@hydra.main(version_base="1.2", config_path=root / "configs", config_name="train.yaml")
def main(config: DictConfig):
    
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.training_pipeline import train
    # Applies optional utilities
    utils.extras(config)
   
    
    # Modifying the configuration
    # config.datamodule._target_ = 'src.datamodules.classification_datamodule.ClassificationDataModule'
    # config.datamodule.dataset_name = 'adult'
    # config.datamodule.data_dir = 'data'
    # config.model._target_ = 'src.models.lightening_module.ClassificationLitModule'
    # config.model.net._target_ = 'src.models.components.models.MLP'
    # config.model.criterion._target_ = 'src.metrics.train_metrics.ClassificationMixedLoss'
    # config.model.calibrator._target_ = 'src.calibration.calibration.TemperatureScaling'
    # config.logger.wandb.project = 'DM_Benchmark'
    # config.trainer.accelerator = 'gpu'
    # config.trainer.devices = '1'
    # Train model
    return train(config)


if __name__ == "__main__":
    sys.argv = [s for s in sys.argv if "--mode" not in s and "--port" not in s]
    main()