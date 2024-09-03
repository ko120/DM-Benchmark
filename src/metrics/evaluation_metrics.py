from typing import Any, List

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import Metric
from torchmetrics import MaxMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.calibration_error import CalibrationError
from src.metrics.train_metrics import ClassificationKernelLoss
from typing import Any, List, Literal, Optional, Dict, Callable

# Evaluating Metrics
class ShannonEntropyError(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("entropy_total", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, is_dist=False):
        p = logits if is_dist else F.softmax(logits, dim=-1)

        self.entropy_total += torch.sum(- p * torch.log(p))
        self.count += logits.shape[0]

    def compute(self):
        return self.entropy_total.float() / self.count.float()

class ClassificationKernelCalibrationError(Metric):
    def __init__(self, dist_sync_on_step=False, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("kcal_total", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.kcal_func = ClassificationKernelLoss(**kwargs)

    def update(self, preds: torch.Tensor, target: torch.Tensor, inputs: torch.Tensor, verbose=False):
        kcal = self.kcal_func(inputs, target, preds, verbose=verbose)

        self.kcal_total += kcal
        self.count += 1

    def compute(self):
        return self.kcal_total.float() / self.count

