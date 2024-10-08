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
        task_criterion: Callable,
        reg_criterion:Callable, 
        calibrator: Optional[Callable] = None,
        lr: float = 0.001,
        weight_decay: float = 0.0005
        ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['net'])  # this is needed for efficiency
        self.net = net
        # check whether adversarial training
        self.is_adv = self.net.adv
        # disable automatic optimization for simulatneous gradient descent for adversarial loss
        if self.is_adv:
            self.automatic_optimization = False
            
        # store outputs
        self.train_outputs ={"logits": [],"sensitive": [], "preds": [], "targets":[]}
        self.val_outputs ={"logits": [], "sensitive": [], "preds": [], "targets":[]}
        self.test_outputs ={"logits": [], "sensitive": [], "preds": [], "targets":[]}
        
        # loss function
        self.task_criterion = task_criterion
        self.reg_criterion= reg_criterion
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

        # For logging best validation metrics
        self.val_acc_best = MaxMetric()
        self.val_ece_best = MinMetric()
        self.val_dp_best = MaxMetric()

        # Additional metrics for post-hoc calibration
        if self.calibrator:
            self.test_calibrated_acc = Accuracy(task= task)
            self.test_calibrated_ece = CalibrationError(**ece_kwargs)
            self.test_calibrated_dp = BinaryFairness(num_groups = 2)

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()
        self.val_ece_best.reset()
        self.val_dp_best.reset()

    def step(self, batch: Any):
        X, Y, A, A_prop = batch
        Y = Y.squeeze(-1)
        A = A.squeeze(-1)
        # forward pass for classifier
        Z = self.net.encoder(X)
        logit =  self.net.classifier(Z)
        with torch.no_grad():
            preds = torch.argmax(logit,dim=-1)

        return X, Y, logit, A, preds, Z, A_prop
    # # This is Gan like implementation, and it takes longer time with same result.
    # def training_step(self, batch: Any, batch_idx: int):
    #     # detect Nan Inf for grad (This is needed since we are taking gradient of L1 Norm)    
    #     torch.autograd.set_detect_anomaly(True)

    #     Y, logits, A, preds, Z, A_prop = self.step(batch)
    #     optimizer, optimizer_d = self.optimizers()
    #     # taking only gradient on encoder + classifier network    
    #     self.toggle_optimizer(optimizer) 

    #     # Loss computation
    #     ce_loss = self.task_criterion(Y, logits)
    #     self.manual_backward(ce_loss,retain_graph=True)
    #     self.clip_gradients(optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
    #     optimizer.step()
    #     optimizer.zero_grad()
    #     # stop tracking gradient for encoder + classifier network 
    #     self.untoggle_optimizer(optimizer)

    #     # taking only gradient on adversarial network 
    #     self.toggle_optimizer(optimizer_d) 

    #     # Loss computation
    #     # forward pass for adversarial
    #     # detaching is needed to prevent gradient flow back to classifier
    #     A_hat = self.net.discriminator(Z.detach().clone()) 
    #     A_hat_prob = torch.sigmoid(A_hat)
    #     A_hat_prob = A_hat_prob.squeeze(-1)
    #     aud_loss = -self.reg_criterion(y=A, logits=A_hat_prob)
    #     # multiplying by proportion of group from dataset to compute DP loss = E_z0[h] + E_z1[1-h]
    #     weights = A_prop[0][0]*(1-A) + A_prop[1][1]*(A)
    #     aud_loss = torch.mean(weights* aud_loss)
    #     self.manual_backward(aud_loss)
    #     self.clip_gradients(optimizer_d, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
    #     optimizer_d.step()
    #     optimizer_d.zero_grad()
    #     # stop tracking gradient for discriminator network 
    #     self.untoggle_optimizer(optimizer_d)

    #     loss = ce_loss+ aud_loss
    
    #     self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
    #     self.log("train/ce_loss", ce_loss, on_step=False, on_epoch=True, prog_bar=True)
    #     self.log("train/aud_loss", aud_loss, on_step=False, on_epoch=True, prog_bar=True)
        
    #     self.train_outputs["sensitive"].append(A)
    #     self.train_outputs["targets"].append(Y)
    #     self.train_outputs["preds"].append(preds)
    #     self.train_outputs["logits"].append(logits)

    #     # remember to always return loss from `training_step()` or else backpropagation will fail!
    #     return loss

    def adv_loss(self, Y, logits, A, preds, Z, A_prop, task_loss, train):
        if train:
            torch.autograd.set_detect_anomaly(True)
            optimizer, optimizer_d = self.optimizers()

            # taking only gradient on encoder + classifier network    
            # Loss computation
            A_hat_prob = self.net.discriminator(Z.detach()) 
            A_hat_prob = A_hat_prob.squeeze(-1)
            reg_loss = -self.reg_criterion(y=A,logits=A_hat_prob)
            # multiplying by proportion of group from dataset to compute DP loss = E_z0[h] + E_z1[1-h]
            weights = A_prop[0][0]*(1-A) + A_prop[1][1]*(A)
            reg_loss = torch.mean(weights * reg_loss)

            loss = self.reg_criterion.loss_scalers * reg_loss + task_loss

            self.toggle_optimizer(optimizer) 
            optimizer.zero_grad()
            self.manual_backward(loss,retain_graph=True)
            self.clip_gradients(optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            optimizer.step()
            
            # stop tracking gradient for encoder + classifier network 
            self.untoggle_optimizer(optimizer)

            # taking only gradient on adversarial network 
            self.toggle_optimizer(optimizer_d) 
            optimizer_d.zero_grad()
            self.manual_backward(reg_loss)
            self.clip_gradients(optimizer_d, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            optimizer_d.step()
            
            # stop tracking gradient for discriminator network 
            self.untoggle_optimizer(optimizer_d)
            return reg_loss, loss
        else: # validation
            # computing loss    
            # forward pass for adversarial
            A_hat_prob = self.net.discriminator(Z.detach().clone())
            A_hat_prob = A_hat_prob.squeeze(-1)
            reg_loss = -self.reg_criterion(y=A,logits=A_hat_prob)
            weights = A_prop[0][0]*(1-A) + A_prop[1][1]*(A)
            reg_loss = torch.mean(weights * reg_loss) 

            return reg_loss 

        

    # paper implementation: https://github.com/charan223/FairDeepLearning/blob/main/models/model_laftr.py
    def training_step(self, batch: Any, batch_idx: int):
        X, Y, logits, A, preds, Z, A_prop  =self.step(batch)
        task_loss = self.task_criterion.loss_scalers * self.task_criterion(Y,logits)
        if self.is_adv:
            reg_loss, loss = self.adv_loss(Y, logits, A, preds, Z, A_prop, task_loss, True)
        else:
            Z0 = Z[A==0]
            Z1 = Z[A==1]
            
            reg_loss = self.reg_criterion.loss_scalers * self.reg_criterion(x= None, y=Z0,logits=Z1)
            loss = task_loss + reg_loss 
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/task_loss", task_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/reg_loss", reg_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.train_outputs["sensitive"].append(A)
        self.train_outputs["targets"].append(Y)
        self.train_outputs["preds"].append(preds)
        self.train_outputs["logits"].append(logits)

        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return loss

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
        X, Y, logits, A, preds, Z, A_prop  =self.step(batch)
        task_loss = self.task_criterion.loss_scalers * self.task_criterion(Y,logits)
        if self.is_adv:
            reg_loss = self.reg_criterion.loss_scalers * self.adv_loss(Y, logits, A, preds, Z, A_prop, task_loss, False)
        else:
            Z0 = Z[A==0]
            Z1 = Z[A==1]
            reg_loss = self.reg_criterion.loss_scalers * self.reg_criterion(x= None, y=Z0,logits=Z1)
        loss = task_loss + reg_loss

        # logging
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/task_loss", task_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/reg_loss", reg_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.val_outputs["sensitive"].append(A)
        self.val_outputs["logits"].append(logits)
        self.val_outputs["targets"].append(Y)
        self.val_outputs["preds"].append(preds)

        return loss

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
        X, Y, logits, A, preds, Z, A_prop  =self.step(batch)
        task_loss = self.task_criterion.loss_scalers * self.task_criterion(Y,logits)
        if self.is_adv:
            reg_loss = self.reg_criterion.loss_scalers * self.adv_loss(Y, logits, A, preds, Z, A_prop, task_loss, False)
        else:
            Z0 = Z[A==0]
            Z1 = Z[A==1]
            reg_loss = self.reg_criterion.loss_scalers * self.reg_criterion(x= None, y=Z0,logits=Z1)
        loss = task_loss + reg_loss
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/task_loss", task_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/reg_loss", reg_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.test_outputs["sensitive"].append(A)
        self.test_outputs["logits"].append(logits)
        self.test_outputs["targets"].append(Y)
        self.test_outputs["preds"].append(preds)

        return loss

    def on_test_epoch_end(self):
        epoch_sensitive = torch.cat(self.test_outputs["sensitive"], dim=0)
        epoch_logits = torch.cat(self.test_outputs["logits"], dim=0)
        epoch_targets = torch.cat(self.test_outputs["targets"], dim=0)
        epoch_preds = torch.cat(self.test_outputs["preds"], dim=0)

        # log test metrics
        acc = self.test_acc(epoch_preds, epoch_targets)
        ece = self.test_ece(epoch_logits, epoch_targets)
        dp_dict = self.test_dp(preds= epoch_preds, target = epoch_targets, groups = epoch_sensitive)
        dp = next(iter(dp_dict.values())).cpu().item()

        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/ece", ece, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/dp", dp, on_step=False, on_epoch=True, prog_bar=True)

        # If post-hoc calibration method is chosen, apply it to model predictions
        if self.calibrator:
            pred_dist = F.softmax(epoch_logits, dim=-1)
            with torch.no_grad():
                calibrated_dists = self.calibrator(pred_dist)
                calibrated_preds = torch.argmax(calibrated_dists, dim=-1)

            calibrated_acc = self.test_calibrated_acc(calibrated_preds, epoch_targets)
            calibrated_ece = self.test_calibrated_ece(calibrated_dists, epoch_targets)
            calib_dp_dict = self.test_calibrated_dp(preds= calibrated_preds, target = epoch_targets, groups = epoch_sensitive)
            calib_dp = next(iter(calib_dp_dict.values())).cpu().item()

            # log post-hoc calibrated test metrics
            self.log(f"test/calibrated_acc", calibrated_acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"test/calibrated_ece ", calibrated_ece , on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"test/calibrated_dp", calib_dp, on_step=False, on_epoch=True, prog_bar=True)
        
        # need to clear at the end
        self.test_outputs["sensitive"].clear()
        self.test_outputs["logits"].clear()
        self.test_outputs["targets"].clear()
        self.test_outputs["preds"].clear()


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


        if self.calibrator:
            self.test_calibrated_acc.reset()
            self.test_calibrated_ece.reset()
            self.test_calibrated_dp.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optim = torch.optim.Adam(params=self.net.predict_params(),lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # if adversarial learning we return optimizer for discriminator
        if self.is_adv:
            optim_disc = torch.optim.Adam(params= self.net.audit_params(),lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
            return optim, optim_disc
        return optim