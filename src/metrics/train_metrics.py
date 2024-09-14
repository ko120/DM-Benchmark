import torch
from torch import nn
import torchmetrics
import torch.nn.functional as F
from typing import Optional, Dict, List
from scipy.special import erf
import numpy as np
import time
import pdb
from geomloss import SamplesLoss

# Kernels Utils
def rbf_kernel(u: torch.Tensor, v: torch.Tensor, bandwidth=1):
    diff_norm_mat = torch.norm(u.unsqueeze(1) - v, dim=2).square()
    return torch.exp(- diff_norm_mat / bandwidth)

def quadrant_partition_kernel(u: torch.Tensor, v: torch.Tensor):
    raise NotImplementedError()


def norm_partition_kernel(u: torch.Tensor, v: torch.Tensor):
    raise NotImplementedError()

def tanh_kernel(u: torch.Tensor, v: torch.Tensor, bandwidth=1):
    out = torch.tanh(v) * torch.tanh(u).unsqueeze(1) # N x N x 1 x num_samples
    return out.squeeze(2)
kernel_funs = {"rbf": rbf_kernel,
               "partition_quadrant": quadrant_partition_kernel,
               "partition_norm": norm_partition_kernel,
               "tanh": tanh_kernel}

VALID_OPERANDS = ['x', 'y', 'p', 'coords']

def mean_no_diag(A):
    assert A.dim() == 2 and A.shape[0] == A.shape[1]
    n = A.shape[0]
    A = A - torch.eye(n).to(A.device) * A.diag()
    return A.sum() / (n * (n - 1))




# Loss
class ClassificationCELoss:
    """
        Cross-entropy loss for classification.
    """
    def __init__(self, loss_scalers=1, **kwargs):
        self.loss = torch.nn.CrossEntropyLoss(**kwargs)
        self.loss_scalers = loss_scalers

    def __call__(self, y, logits):
        return self.loss(logits, y) * self.loss_scalers

class ClassificationKernelLoss:
    """
        MMD loss function for classification tasks.
        Allows for distribution matching by specifying operands and kernel functions.
        `scalers` and `bandwidths` are the parameters of the kernel functions.
        It requires output dim=2 for binary case
    """
    def __init__(self,
                 loss_scalers: 0.2, 
                 operands: Dict[str, str] = {'x': "rbf", 'y': "rbf"},
                 scalers: Optional[Dict] = None,
                 bandwidths: Optional[Dict] = {'x': 0.01, 'y': 1.0}):

        assert all([op in VALID_OPERANDS for op in operands.keys()])

        if scalers is None:
            scalers = {op: 1. for op in operands.keys()}
        else:
            assert all(op in scalers for op in operands.keys())

        self.kernel_fun = {op: kernel_funs[kernel] for op, kernel in operands.items()}
        self.operands = list(operands.keys())
        self.scalers = scalers
        self.bandwidths = bandwidths
        self.loss_scalers = loss_scalers

    def __call__(self, x, y, logits, verbose=False):
        kernel_out = None
        loss_mats = [None for i in range(3)]
        for op in self.operands:
            scaler = self.scalers[op]
            bandwidth = self.bandwidths[op]
            if op == 'x':
                # This is only true for tabular data. For example, multi-channel images will have 4D batches for x.
                assert x.dim() == 2
                # Computing k(x,x)
                loss_mat = loss_mat2 = loss_mat3 = scaler * self.kernel_fun[op](x, x, bandwidth)
            elif op == 'y':
                # Computes MMD loss for classification (See Section 4.1 of paper)
                # Computing Q^2
                num_classes = logits.shape[-1] # we consider num_classes=2 for binary
                y_all = torch.eye(num_classes).to(logits.device)
                k_yy = self.kernel_fun[op](y_all, y_all, bandwidth)
                q_y = F.softmax(logits, dim=-1)
                q_yy = torch.einsum('ic,jd->ijcd', q_y, q_y)
                total_yy = q_yy * k_yy.unsqueeze(0)
                # Computing PQ
                k_yj = k_yy[:,y].T
                total_yj = torch.einsum('ic,jc->ijc', q_y, k_yj)
                y_one_hot = F.one_hot(y, num_classes=num_classes).float()
                # Computing P^2
                loss_mat = scaler * total_yy.sum(dim=(2,3))
                loss_mat2 = scaler * total_yj.sum(-1)
                loss_mat3 = scaler * self.kernel_fun[op](y_one_hot, y_one_hot, bandwidth)
            else:
                assert False, f"When running classification, operands must be x and y. Got operand {op} instead."
            # Computing Expectations
            for i, value in enumerate([loss_mat, loss_mat2, loss_mat3]):
                if loss_mats[i] is None:
                    loss_mats[i] = value
                else:
                    loss_mats[i] =  loss_mats[i] * value
        # MMD = E_Q[k(x,x)]^2 -2E_P[E_Q[k(x,x)]] + E_P[k(x,x)]^2, we are ignoring diagonal since it is kernel distance by itself
        kernel_out = mean_no_diag(loss_mats[0]) - 2 * mean_no_diag(loss_mats[1]) + mean_no_diag(loss_mats[2])

        return kernel_out * self.loss_scalers

# need to ensure that input for this loss function should include x due to consistency with MMD
class ClassificationL1Loss:
    """
        L1 loss for classification
    """
    def __init__(self, loss_scalers=1, **kwargs):
        self.loss = torchmetrics.MeanAbsoluteError(**kwargs)
        self.loss_scalers = loss_scalers
            
    def __call__(self, x=None, y=None, logits=None):
        self.loss.to(logits.device)
        return self.loss(logits, y) * self.loss_scalers

# class ClassificationSinkhornLoss:
#     """
#         Sinkhorn Loss for classification
#     """
#     def __init__(self, loss_scalers=1, **kwargs):
#         self.loss = SamplesLoss("sinkhorn", **kwargs)
#         self.loss_scalers = loss_scalers
            
#     def __call__(self, x=None, y=None, logits=None):
#         self.loss.to(logits.device)
#         prob_logits = F.softmax(logits, dim=1)
#         num_classes = logits.size(-1)
#         y_one_hot = F.one_hot(y, num_classes=num_classes).float()
#         return self.loss(prob_logits,y_one_hot) * self.loss_scalers

class ClassificationSinkhornLoss:
    """
        Sinkhorn Loss for classification
    """
    def __init__(self, loss_scalers=1, **kwargs):
        # self.loss = SamplesLoss("sinkhorn", **kwargs)
        self.loss = SamplesLoss(loss="sinkhorn")
        self.loss_scalers = loss_scalers
            
    def __call__(self, x=None, y=None, logits=None):
        self.loss.to(logits.device)
        epsilon = 1e-8  # Small value to avoid division by zero
        # taking argmax for logit then add dim to match N,D format
        pred_classes = torch.argmax(logits, dim=-1).float()
        pred_classes = pred_classes.view(-1, 1)
        # normalize it to adjust to have equal mass
        pred_classes_normalized = pred_classes / (pred_classes.sum() + epsilon)
        y = y.view(-1, 1).float()
        y_normalized = y / (y.sum() + epsilon)
        return self.loss(pred_classes_normalized,y_normalized) * self.loss_scalers

# class ClassificationMixedLoss:
#     """
#         Mixed loss function (MMD + NLL) for classification.
#         `loss_scalers` determines the mixture weight between MMD and NLL.
#     """
#     def __init__(self, reg_type = "mmd", loss_scalers: Optional[Dict] = None, **kwargs):
#         if loss_scalers is None:
#             loss_scalers = {"nll": 1, "mmd": 1, "sink": 1}
#         else:
#             assert set(loss_scalers.keys()) == {"nll", "mmd","sink"}
#         self.reg_type = reg_type
#         self.loss_scalers = loss_scalers
#         self.nll = torch.nn.CrossEntropyLoss()
#         self.mmd = ClassificationKernelLoss()
#         self.sinkhorn = ClassificationSinkhornLoss()
#         self.reg_dict = {"mmd":self.mmd, "sinkhorn":self.sinkhorn}
#         self.reg = self.loss_dict[reg_type]

#     def __call__(self, x, y, logits):
#         if self.reg_type == "sink":
#             reg = self.mmd(x,y,logits)
#         elif self
     
#          * self.mmd(x, y, logits)
#         return self.loss_scalers["nll"] * self.nll(logits, y) 

