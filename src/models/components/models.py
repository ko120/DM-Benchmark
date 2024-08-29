from torch import nn
import torch.nn.functional as F
import torch


class SimpleDenseNet(nn.Module):
    """
        Neural network model for classificaiton.
        Forward() call returns logits.
    """
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        if use_batchnorm:
            self.model = nn.Sequential(
                nn.Linear(input_size, lin1_size),
                nn.BatchNorm1d(lin1_size),
                nn.ReLU(),
                nn.Linear(lin1_size, lin2_size),
                nn.BatchNorm1d(lin2_size),
                nn.ReLU(),
                nn.Linear(lin2_size, lin3_size),
                nn.BatchNorm1d(lin3_size),
                nn.ReLU(),
                nn.Linear(lin3_size, output_size),
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(input_size, lin1_size),
                nn.ReLU(),
                nn.Linear(lin1_size, lin2_size),
                nn.ReLU(),
                nn.Linear(lin2_size, lin3_size),
                nn.ReLU(),
                nn.Linear(lin3_size, output_size),
            )
        self.output_size = output_size

    def forward(self, x):
        return self.model(x)

        


class MLP(nn.Module):
    """
    MLP layer with declaring number of neurons as list ex. [input_size] + hdepth*[hwdith] +[output_size]
    """
    def __init__(self, input_size, hwdith, hdepth, output_size, activ="leakyrelu"):
        """Initializes MLP unit"""
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layers = [input_size] + hdepth*[hwdith] +[output_size] # output size becomes 1 for binary
        self.num_layers = len(self.layers) - 1
        self.hiddens = nn.ModuleList(
            [
                nn.Linear(self.layers[i], self.layers[i + 1])
                for i in range(self.num_layers)
            ]
        )
        for hidden in self.hiddens:
            torch.nn.init.xavier_uniform_(hidden.weight)
        self.activ = activ

    def forward(self, inputs):
        """Computes forward pass through the model"""
        L = inputs
        for hidden in self.hiddens:
            L = hidden(L)
            if self.activ == "softplus":
                L = F.softplus(L)
            elif self.activ == "sigmoid":
                L = F.sigmoid(L)
            elif self.activ == "relu":
                L = F.relu(L)
            elif self.activ == "leakyrelu":
                L = F.leaky_relu(L)
            elif self.activ == "None":
                pass
            else:
                raise Exception("bad activation function")
        return L

    # def freeze(self):
    #     """Stops gradient computation through MLP parameters"""
    #     for para in self.parameters():
    #         para.requires_grad = False

    # def activate(self):
    #     """Activates gradient computation through MLP parameters"""
    #     for para in self.parameters():
    #         para.requires_grad = True


class LaftrNet(nn.Module):
    def __init__(self, input_size ,output_size ,zdim , edepth, ewidths, cdepth, cwidths, adepth, awidths, num_groups):
        super(LaftrNet,self).__init__()
        # for the adversary network, we make output class=1
        # declare models
        self.encoder = MLP(input_size = input_size, hdepth = edepth,hwdith = ewidths, output_size = zdim)
        self.classifier = MLP(input_size = zdim, hdepth = cdepth, hwdith = cwidths, output_size = output_size)
        self.discriminator = MLP(input_size = zdim, hdepth = adepth, hwdith = awidths, output_size = num_groups-1)
        self.output_size = output_size


    def forward(self,x):
        return self.encoder(x)
        
    def predict_params(self):
        """Returns encoder and classifier parameters"""
        return list(self.classifier.parameters()) + list(self.encoder.parameters())

    def audit_params(self):
        """Returns discriminator parameters"""
        return self.discriminator.parameters()

        
