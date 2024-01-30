
import torch
from torch.nn import Module


class CBDNetwork(Module):
    def __init__(self, noise_predictor: Module,
                 reconstruction_network: Module):
        super(CBDNetwork, self).__init__()
        self.noise_predictor = noise_predictor
        self.reconstruction_network = reconstruction_network

    def forward(self, x: torch.Tensor):
        """Forward pass for the network.

        :return: A torch.Tensor.
        """

        # raise NotImplementedError

        return self.reconstruction_network(x)