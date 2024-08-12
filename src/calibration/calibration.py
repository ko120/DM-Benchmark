
import numpy as np
import itertools
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import os, sys, shutil, copy, time, random


# Calibration
def _get_prediction_device(predictions):
    """ Get the device of a prediction

    Args:
        predictions: a prediction of any type.

    Returns:
        device: the torch device that prediction is on.
    """
    if issubclass(type(predictions), torch.distributions.distribution.Distribution):
        with torch.no_grad():
            device = predictions.sample().device    # Trick to get the device of a torch Distribution class because there is no interface for this
    elif issubclass(type(predictions), dict):
        assert len(predictions.keys()) != 0, "Must have at least one element in the ensemble"
        device = _get_prediction_device(predictions[next(iter(predictions))])   # Return the device of the first element in the dictionary
    else:
        device = predictions.device
    return device

class Calibrator:
    """ The abstract base class for all calibrator classes.

    Args:
        input_type (str): the input prediction type.
            If input_type is 'auto' then it is automatically induced when Calibrator.train() or update() is called, it cannot be changed after the first call to train() or update().
            Not all sub-classes support 'auto' input_type, so it is strongly recommended to explicitly specify the prediction type.
    """
    def __init__(self, input_type='auto'):
        self.input_type = input_type
        self.device = None

    def _change_device(self, predictions):
        """ Move everything into the same device as predictions, do nothing if they are already on the same device """
        # print("_change_device is deprecated ")
        device = _get_prediction_device(predictions)
        # device = self.get_device(predictions)
        self.to(device)
        self.device = device
        return device


    def to(self, device):
        """ Move this class and all the tensors it owns to a specified device.

        Args:
            device (torch.device): the device to move this class to.
        """
        assert False, "Calibrator.to has not been implemented"


    def train(self, predictions, labels, *args, **kwargs):
        """ The train abstract class. Learn the recalibration map based on labeled data.

        This function uses the training data to learn any parameters that is necessary to transform a low quality (e.g. uncalibrated) prediction into a higher quality (e.g. calibrated) prediction.
        It takes as input a set of predictions and the corresponding labels.
        In addition, a few recalibration algorithms --- such as group calibration or multicalibration --- can take as input additional side features, and the transformation depends on the side feature.

        Args:
            predictions (object): a batched prediction object, must match the input_type argument when calling __init__.
            labels (tensor): the labels with shape [batch_size]
            side_feature (tensor): some calibrator instantiations can use additional side feature, when used it should be a tensor of shape [batch_size, n_features]

        Returns:
            object: an optional log object that contains information about training history.
        """
        assert False, "Calibrator.train has not been implemented"

    #
    # If half_life is not None, then it is the number of calls to this function where the sample is discounted to 1/2 weight
    # Not all calibration functions support half_life
    def update(self, predictions, labels, *args, **kwargs):
        """ Same as Calibrator.train, but updates the calibrator online with the new data (while train erases any existing data in the calibrator and learns it from scratch)

        Args:
            predictions (object): a batched prediction object, must match the input_type argument when calling __init__.
            labels (tensor): the labels with shape [batch_size]
            side_feature (tensor): some calibrator instantiations can use additional side feature, when used it should be a tensor of shape [batch_size, n_features]

        Returns:
            object: an optional log object that contains information about training history.
        """
        assert False, "Calibrator.update has not been implemented"

    # Input an array of shape [batch_size, num_classes], output the recalibrated array
    # predictions should be in the same pytorch device
    # If side_feature is not None when calling train, it shouldn't be None here either.
    def __call__(self, predictions, *args, **kwargs):
        """ Use the learned calibrator to transform new data.

        Args:
            predictions (prediction object): a batched prediction object, must match the input_type argument when calling __init__.
            labels (tensor): the labels with shape [batch_size]
            side_feature (tensor): some calibrator instantiations can use additional side feature, when used it should be a tensor of shape [batch_size, n_features]

        Returns:
            prediction object: the transformed predictions
        """
        assert False, "Calibrator.__call__ has not been implemented"

    def check_type(self, predictions):
        """ Checks that the prediction has the correct shape specified by input_type.

        Args:
            predictions (prediction object): a batched prediction object, must match the input_type argument when calling __init__.
        """
        if self.input_type == 'point':
            assert len(predictions.shape) == 1, "Point prediction should have shape [batch_size]"
        elif self.input_type == 'interval':
            assert len(predictions.shape) == 2 and predictions.shape[1] == 2, "interval predictions should have shape [batch_size, 2]"
        elif self.input_type == 'quantile':
            assert len(predictions.shape) == 2 or (len(predictions.shape) == 3 and predictions.shape[2] == 2), "quantile predictions should have shape [batch_size, num_quantile] or [batch_size, num_quantile, 2]"
        elif self.input_type == 'distribution':
            # assert hasattr(predictions, 'cdf') and hasattr(predictions, 'icdf'), "Distribution predictions should have a cdf and icdf method"
            assert hasattr(predictions, 'cdf') , "Distribution predictions should have a cdf method"

    def assert_type(self, input_type, valid_types):
        msg = "Input data type not supported, input data type is %s, supported types are %s" % (input_type, " ".join(valid_types))
        assert input_type in valid_types, msg




class TemperatureScaling(Calibrator):
    """ The class to recalibrate a categorical prediction with temperature scaling

    Temeprature scaling is often the algorithm of choice when calibrating predictions from deep neural networks.
    The only learnable parameter --- the temperature parameter $T$ --- is tuned to maximize the log-likelihood of the labels.
    Temperature scaling requires very few samples to train because it only learns a single parameter $T$, yet despite the simplcity,
    empirical results show that temperature scaling achieves low calibration error when applied to deep network predictions.

    Args:
        verbose (bool): if verbose=True print detailed messsages
    """
    def __init__(self, verbose=False):
        super(TemperatureScaling, self).__init__(input_type='categorical')
        self.verbose = verbose
        self.temperature = None

    def train(self, predictions, labels, *args, **kwargs):
        """ Find the optimal temperature with gradient descent.

        Args:
            predictions (tensor): a batch of categorical predictions with shape [batch_size, num_classes]
            labels (tensor): a batch of labels with shape [batch_size]
        """
        # Use gradient descent to find the optimal temperature
        # Can add bisection option in the future, since it should be considerably faster
        self.to(predictions)

        self.temperature = torch.ones(1, 1, requires_grad=True, device=self.device)
        optim = torch.optim.Adam([self.temperature], lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', patience=3, threshold=1e-6, factor=0.5)

        log_prediction = torch.log(predictions + 1e-10).detach()

        # Iterate at most 100k iterations, but expect to stop early
        for iteration in range(100000):
            optim.zero_grad()
            adjusted_predictions = log_prediction / self.temperature
            loss = F.cross_entropy(adjusted_predictions, labels)
            loss.backward()
            optim.step()
            lr_scheduler.step(loss)

            # Hitchhike the lr scheduler to terminate if no progress
            if optim.param_groups[0]['lr'] < 1e-6:
                break
            if self.verbose and iteration % 100 == 0:
                print("Iteration %d, lr=%.5f, NLL=%.3f" % (iteration, optim.param_groups[0]['lr'], loss.cpu().item()))

    def __call__(self, predictions, *args, **kwargs):
        """ Use the learned temperature to calibrate the predictions.

        Only use this after calling TemperatureScaling.train.

        Args:
            predictions (tensor): a batch of categorical predictions with shape [batch_size, num_classes]

        Returns:
            tensor: the calibrated categorical prediction, it should have the same shape as the input predictions
        """
        if self.temperature is None:
            print("Error: need to first train before calling this function")
        self.to(predictions)
        log_prediction = torch.log(predictions + 1e-10)
        return torch.softmax(log_prediction / self.temperature, dim=1)

    def to(self, device):
        """ Move all assets of this class to a torch device.

        Args:
            device (device): the torch device (such as torch.device('cpu'))
        """
        device = _get_prediction_device(device)
        if self.temperature is not None:
            self.temperature.to(device)
        self.device = device
        return self