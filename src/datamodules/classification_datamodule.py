# Data Module
from typing import Optional, Tuple
import os
import pandas as pd
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from functools import partial

class ClassificationDataModule(LightningDataModule):
    """Datamodule for classification datasets.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        data_dir: str = "data/",
        random_seed: Optional[int] = 1,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.1, 0.2),
        batch_size: int = 64,
        test_batch_size: int = 128,
        normalize: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False
    ):
        super().__init__()

        assert np.isclose(sum(train_val_test_split), 1), f"Train_val_test_split must sum to 1. Got {train_val_test_split} with sum {sum(train_val_test_split):0.5f}."

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.input_size: int = classification_shapes[dataset_name][0]
        self.label_size: int = classification_shapes[dataset_name][1]

        self.setup()

    @property
    def dataset_type(self) -> str:
        return "regression"

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            loader_fun = classification_load_funs[self.hparams.dataset_name]
            X, y = loader_fun(self.hparams.data_dir)
            if self.hparams.normalize:
                std = X.std(axis=0)
                zeros = np.isclose(std, 0.)
                X[:, ~zeros] = (X[:, ~zeros] - X[:, ~zeros].mean(axis=0)) / X[:, ~zeros].std(axis=0)
                X[:, zeros] = 0.
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            # Split based on initialized ratio
            dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y).long())
            lengths = [int(len(X) * p) for p in self.hparams.train_val_test_split]
            lengths[-1] += len(X) - sum(lengths)  # fix any rounding errors
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=lengths,
                generator=torch.Generator().manual_seed(self.hparams.random_seed),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.test_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.test_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True
        )

def _load_adult(data_dir):
    """
    Attribute Information:
    The dataset contains 16 columns
    Target filed: Income
    -- The income is divide into two classes: <=50K and >50K
    Number of attributes: 14
    -- These are the demographics and other features to describe a person
    """
    data_file = os.path.join(data_dir, 'classification/adult/adult.data')
    colnames = ["age","workclass","fnlwgt","education","educational-num","marital-status","occupation","relationship","race","gender","capital-gain","capital-loss","hours-per-week","native-country","income"]
    data = pd.read_csv(data_file, header=None, names=colnames, skipinitialspace=True)
    data = data.replace("?", np.nan).dropna()
    category_col =['workclass', 'education','marital-status', 'occupation',
                  'relationship', 'race', 'gender', 'native-country']
    b, c = np.unique(data['income'], return_inverse=True)
    data['income'] = c # turn into binary [0,1]

    def encode_and_bind(original_dataframe, feature_to_encode):
      dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
      res = pd.concat([original_dataframe, dummies], axis=1)
      res = res.drop([feature_to_encode], axis=1)
      return res

    for feature in category_col:
        data = encode_and_bind(data, feature)

    y = data['income'].to_numpy()
    data = data.drop('income', axis=1)
    X = data.to_numpy().astype(float)
    return X, y


classification_load_funs = {
    "adult": _load_adult}

classification_shapes = {
    "wdbc": (30, 2),
    "adult": (104, 2),
    "heart-disease": (23, 5),
    "online-shoppers": (28, 2),
    "dry-bean": (16, 7)
}
