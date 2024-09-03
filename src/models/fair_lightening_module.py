from typing import Any, List

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import torchmetrics
from torchmetrics import MaxMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.calibration_error import CalibrationError
from torchmetrics.classification import BinaryFairness
import pdb
from typing import Any, List, Literal, Optional, Dict, Callable
from src.metrics import ShannonEntropyError, ClassificationKernelCalibrationError
# Core NN Module
class fair_ClassificationLitModule(LightningModule):
    """ LightningModule for Classification tasks.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        criterion: Callable,
        criterion2:Callable = None, 
        calibrator: Callable = None,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        kcal_kwargs = None
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['net'])  # this is needed for efficiency
        self.net = net
        # disable automatic optimization for simulatneous gradient descent
        self.automatic_optimization = False
        # store outputs
        self.train_outputs ={"logits": [],"sensitive": [], "preds": [], "targets":[]}
        self.val_outputs ={"logits": [], "sensitive": [], "preds": [], "targets":[]}
        self.test_outputs ={"logits": [], "sensitive": [], "preds": [], "targets":[], "x":[]}
        
        # loss function
        self.criterion = criterion
        self.criterion2= criterion2
        # self.criterion = criterion
        self.calibrator = calibrator
        task = 'binary' if net.output_size == 2 else 'multiclass'

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy(task= task)
        self.val_acc = Accuracy(task= task)
        self.test_acc = Accuracy(task= task)
        # Initialize metrics
        # Even for binary case, we need output_size=2 for computing MMD
        assert net.output_size >= 2, f"Must have >=2 classes for classification task. Model only has {net.output_size} classes." 
        ece_kwargs = {"task": 'multiclass', "n_bins": 20, "norm": 'l1', "num_classes": net.output_size} # We are always using Multiclass ECE since we are considering binary case as multiclass by 0 as class1 and 1 as class 2
        self.train_ece = CalibrationError(**ece_kwargs)
        self.val_ece = CalibrationError(**ece_kwargs)
        self.test_ece = CalibrationError(**ece_kwargs)

        # Torchmetric uses min pos_rate/ max pos_rate, so close to 1 is better
        self.train_dp = BinaryFairness(num_groups= self.net.num_groups)
        self.val_dp = BinaryFairness(num_groups= self.net.num_groups)
        self.test_dp = BinaryFairness(num_groups= self.net.num_groups)

        self.train_entropy = ShannonEntropyError()
        self.val_entropy = ShannonEntropyError()
        self.test_entropy = ShannonEntropyError()

        if kcal_kwargs is None:
            kcal_kwargs = {}
        self.test_kcal = ClassificationKernelCalibrationError(**kcal_kwargs)

        # For logging best validation metrics
        self.val_acc_best = MaxMetric()
        self.val_ece_best = MinMetric()
        self.val_entropy_best = MinMetric()
        self.val_dp_best = MaxMetric()

        # Additional metrics for post-hoc calibration
        if self.calibrator:
            self.test_calibrated_acc = Accuracy(task= task)
            self.test_calibrated_ece = CalibrationError(**ece_kwargs)
            self.test_calibrated_entropy = ShannonEntropyError()
            self.test_calibrated_kcal = ClassificationKernelCalibrationError(**kcal_kwargs)

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()
        self.val_ece_best.reset()
        self.val_entropy_best.reset()
        self.val_dp_best.reset()

    def step(self, batch: Any):
        X, Y, A, A_prop = batch
        Y = Y.squeeze(-1)
        A = A.squeeze(-1)
        # forward pass for classifier
        Z = self.net.encoder(X)
        logit =  self.net.classifier(Z)
        preds = torch.argmax(logit,dim=-1)

        return Y, logit, A, preds, Z, A_prop
    def training_step(self, batch: Any, batch_idx: int):
        # detect Nan Inf for grad (This is needed since we are taking gradient of L1 Norm)    
        torch.autograd.set_detect_anomaly(True)

        Y, logits, A, preds, Z, A_prop = self.step(batch)
        optimizer, optimizer_d = self.optimizers()
        # taking only gradient on encoder + classifier network    
        self.toggle_optimizer(optimizer) 

        # Loss computation
        ce_loss = self.criterion(logits, Y)
        self.manual_backward(ce_loss,retain_graph=True)
        self.clip_gradients(optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        optimizer.step()
        optimizer.zero_grad()
        # stop tracking gradient for encoder + classifier network 
        self.untoggle_optimizer(optimizer)

        # taking only gradient on adversarial network 
        self.toggle_optimizer(optimizer_d) 

        # Loss computation
        # forward pass for adversarial
        # detaching is needed to prevent gradient flow back to classifier
        A_hat = self.net.discriminator(Z.detach()) 
        A_hat_prob = torch.sigmoid(A_hat)
        A_hat_prob = A_hat_prob.squeeze(-1)
        aud_loss = -self.criterion2(A_hat_prob,A)
        # multiplying by proportion of group from dataset to compute DP loss = E_z0[h] + E_z1[1-h]
        weights = A_prop[0][0]*(1-A) + A_prop[1][1]*(A)
        aud_loss = torch.mean(weights* aud_loss)
        self.manual_backward(aud_loss)
        self.clip_gradients(optimizer_d, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        optimizer_d.step()
        optimizer_d.zero_grad()
        # stop tracking gradient for discriminator network 
        self.untoggle_optimizer(optimizer_d)

        total_loss = ce_loss+ aud_loss
    
        self.log("train/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/ce_loss", ce_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/aud_loss", aud_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.train_outputs["sensitive"].append(A)
        self.train_outputs["targets"].append(Y)
        self.train_outputs["preds"].append(preds)
        self.train_outputs["logits"].append(logits)

        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return total_loss

    # # paper implementation: https://github.com/charan223/FairDeepLearning/blob/main/models/model_laftr.py
    # def training_step(self, batch: Any, batch_idx: int):
    #     # detect Nan Inf for grad (This is needed since we are taking gradient of L1 Norm)    
    #     torch.autograd.set_detect_anomaly(True)

    #     Y, logits, A, preds, Z, A_prop = self.step(batch)
    #     optimizer, optimizer_d = self.optimizers()
    #     # taking only gradient on encoder + classifier network    
        

    #     # Loss computation
    #     ce_loss = self.criterion(logits, Y)
    #     A_hat = self.net.discriminator(Z.detach()) 
    #     A_hat_prob = torch.sigmoid(A_hat)
    #     A_hat_prob = A_hat_prob.squeeze(-1)
    #     aud_loss = -self.criterion2(A_hat_prob,A)
    #     # multiplying by proportion of group from dataset to compute DP loss = E_z0[h] + E_z1[1-h]
    #     weights = A_prop[0][0]*(1-A) + A_prop[1][1]*(A)
    #     aud_loss = torch.mean(weights * aud_loss)

    #     total_loss = aud_loss + ce_loss

    #     self.toggle_optimizer(optimizer) 
    #     optimizer.zero_grad()
    #     self.manual_backward(total_loss,retain_graph=True)
    #     self.clip_gradients(optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
    #     optimizer.step()
        
    #     # stop tracking gradient for encoder + classifier network 
    #     self.untoggle_optimizer(optimizer)

    #     # taking only gradient on adversarial network 
    #     self.toggle_optimizer(optimizer_d) 
    #     optimizer_d.zero_grad()
    #     self.manual_backward(aud_loss)
    #     self.clip_gradients(optimizer_d, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
    #     optimizer_d.step()
        
    #     # stop tracking gradient for discriminator network 
    #     self.untoggle_optimizer(optimizer_d)
        
    #     self.log("train/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
    #     self.log("train/ce_loss", ce_loss, on_step=False, on_epoch=True, prog_bar=True)
    #     self.log("train/aud_loss", aud_loss, on_step=False, on_epoch=True, prog_bar=True)
        
    #     self.train_outputs["sensitive"].append(A)
    #     self.train_outputs["targets"].append(Y)
    #     self.train_outputs["preds"].append(preds)
    #     self.train_outputs["logits"].append(logits)

    #     # remember to always return loss from `training_step()` or else backpropagation will fail!
    #     return total_loss

    def on_train_epoch_end(self):
        # We need to compute and log metrics in this phase. If we compute metrics during training step, every epoch which is computationally ineficient
        epoch_sensitive = torch.cat(self.train_outputs["sensitive"], dim=0)
        epoch_targets = torch.cat(self.train_outputs["targets"], dim=0)
        epoch_preds = torch.cat(self.train_outputs["preds"], dim=0)
        epoch_logits = torch.cat(self.train_outputs["logits"], dim=0)
        
        # log train metrics
        acc = self.train_acc(epoch_preds, epoch_targets)
        ece = self.train_ece(epoch_logits, epoch_targets)
        dp_dict = self.train_dp(preds= epoch_preds, target = epoch_targets, groups = epoch_sensitive)
        dp = next(iter(dp_dict.values())).cpu().item()

        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/ece", ece, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/dp", dp, on_step=False, on_epoch=True, prog_bar=True)

        # need to clear at the end
        self.train_outputs["logits"].clear()
        self.train_outputs["targets"].clear()
        self.train_outputs["preds"].clear()
        self.train_outputs["sensitive"].clear()

    def validation_step(self, batch: Any, batch_idx: int):
        Y, logits, A, preds, Z, A_prop = self.step(batch)
       
        # computing loss    
        ce_loss = self.criterion(logits,Y)
        # forward pass for adversarial
        A_hat = self.net.discriminator(Z.detach())
        A_hat_prob = torch.sigmoid(A_hat)
        A_hat_prob = A_hat_prob.squeeze(-1)
        aud_loss = -self.criterion2(A_hat_prob,A)
        weights = A_prop[0][0]*(1-A) + A_prop[1][1]*(A)
        aud_loss = torch.mean(weights* aud_loss)
        total_loss = ce_loss + aud_loss

        # logging
        self.log("val/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ce_loss", ce_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/aud_loss", aud_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.val_outputs["sensitive"].append(A)
        self.val_outputs["logits"].append(logits)
        self.val_outputs["targets"].append(Y)
        self.val_outputs["preds"].append(preds)

        return 

    def on_validation_epoch_end(self):
        epoch_logits = torch.cat(self.val_outputs["logits"], dim=0)
        epoch_targets = torch.cat(self.val_outputs["targets"], dim=0)
        epoch_preds = torch.cat(self.val_outputs["preds"], dim=0)
        epoch_sensitive = torch.cat(self.val_outputs["sensitive"], dim=0)

        # log val metrics
        acc = self.val_acc(epoch_preds, epoch_targets)
        ece = self.val_ece(epoch_logits, epoch_targets)
        dp_dict = self.val_dp(preds= epoch_preds, target = epoch_targets, groups = epoch_sensitive)
        dp = next(iter(dp_dict.values()))

        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ece", ece, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/dp", dp, on_step=False, on_epoch=True, prog_bar=True)

        # log best metric
        self.val_acc_best(acc)
        self.val_ece_best(ece)
        self.val_dp_best(dp)

        # log `*_best` metrics as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/ece_best", self.val_ece_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/dp_best", self.val_dp_best.compute(), sync_dist=True, prog_bar=True)

        # need to clear at the end
        self.val_outputs["logits"].clear()
        self.val_outputs["targets"].clear()
        self.val_outputs["preds"].clear()
        self.val_outputs["sensitive"].clear()

    def on_test_epoch_start(self):
        if self.calibrator is None:
            return

        val_x, val_y = self.trainer.datamodule.data_val[:]
        val_x, val_y = val_x.to(self.device), val_y.squeeze(-1).to(self.device)

        with torch.no_grad():
            logits = self.forward(val_x)
            val_pred = F.softmax(logits, dim=-1)

        torch.set_grad_enabled(True) # need to set it to True since Lightning by default, it doesn't require grad for val/test
        self.calibrator.train(val_pred, val_y)

    def test_step(self, batch: Any, batch_idx: int):
        Y, logits, A, preds, Z, A_prop = self.step(batch)

        # computing loss
        ce_loss = self.criterion(logits,Y)
        # forward pass for adversarial
        A_hat = self.net.discriminator(Z.detach())
        A_hat_prob = torch.sigmoid(A_hat)
        A_hat_prob = A_hat_prob.squeeze(-1)
        aud_loss = -self.criterion2(A_hat_prob,A)
        weights = A_prop[0][0]*(1-A) + A_prop[1][1]*(A)
        aud_loss = torch.mean(weights* aud_loss)
        total_loss = ce_loss + aud_loss
        
        self.log("test/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/ce_loss", ce_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/aud_loss", aud_loss, on_step=False, on_epoch=True, prog_bar=True)
        

        x,_,_,_ = batch
        self.test_outputs["sensitive"].append(A)
        self.test_outputs["logits"].append(logits)
        self.test_outputs["targets"].append(Y)
        self.test_outputs["preds"].append(preds)
        self.test_outputs["x"].append(x)

        return total_loss

    def on_test_epoch_end(self):
        epoch_sensitive = torch.cat(self.test_outputs["sensitive"], dim=0)
        epoch_logits = torch.cat(self.test_outputs["logits"], dim=0)
        epoch_targets = torch.cat(self.test_outputs["targets"], dim=0)
        epoch_preds = torch.cat(self.test_outputs["preds"], dim=0)
        epoch_x = torch.cat(self.test_outputs["x"], dim=0)

        # log test metrics
        acc = self.test_acc(epoch_preds, epoch_targets)
        ece = self.test_ece(epoch_logits, epoch_targets)
        dp_dict = self.test_dp(preds= epoch_preds, target = epoch_targets, groups = epoch_sensitive)
        dp = next(iter(dp_dict.values())).cpu().item()
        kcal = self.test_kcal(epoch_logits, epoch_targets, epoch_x)

        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/ece", ece, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/kcal", kcal, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/dp", dp, on_step=False, on_epoch=True, prog_bar=True)

        # If post-hoc calibration method is chosen, apply it to model predictions
        if self.calibrator:
            pred_dist = F.softmax(epoch_logits, dim=-1)
            with torch.no_grad():
                calibrated_dists = self.calibrator(pred_dist)
                calibrated_preds = torch.argmax(calibrated_dists, dim=-1)

            calibrated_acc = self.test_calibrated_acc(calibrated_preds, epoch_targets)
            calibrated_ece = self.test_calibrated_ece(calibrated_dists, epoch_targets)
            calibrated_entropy = self.test_calibrated_entropy(calibrated_dists, is_dist=True)
            calibrated_kcal = self.test_kcal(calibrated_dists, epoch_targets, epoch_x)

            # log post-hoc calibrated test metrics
            self.log(f"test/calibrated_acc", calibrated_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"test/calibrated_ece ", calibrated_ece , on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"test/calibrated_entropy", calibrated_entropy, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"test/calibrated_kcal", calibrated_kcal, on_step=False, on_epoch=True, prog_bar=True)
        
        # need to clear at the end
        self.test_outputs["sensitive"].clear()
        self.test_outputs["logits"].clear()
        self.test_outputs["targets"].clear()
        self.test_outputs["preds"].clear()
        self.test_outputs["x"].clear()


    def on_epoch_end(self):
        # Reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

        self.train_dp.reset()
        self.test_dp.reset()
        self.val_dp.reset()

        self.train_ece.reset()
        self.test_ece.reset()
        self.val_ece.reset()

        self.test_kcal.reset()

        if self.calibrator:
            self.test_calibrated_acc.reset()
            self.test_calibrated_ece.reset()
            self.test_calibrated_entropy.reset()
            self.test_calibrated_kcal.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optim = torch.optim.Adam(params=self.net.predict_params(),lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        optim_disc = torch.optim.Adam(params= self.net.audit_params(),lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return [optim, optim_disc], []