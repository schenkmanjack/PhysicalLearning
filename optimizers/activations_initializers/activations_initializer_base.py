from abc import ABC, abstractmethod
import torch

class ActivationsInitializerBase(ABC):

    @staticmethod
    @abstractmethod
    def initialize_u(x, dims):
        """For each hidden layer and the output layer initialize the activations.
        Arguments:
            x (torch.Tensor): The input to the network.
        Returns:
            list: A list of the initialized activations.
        """
        device = x.device
        batch_size = x.size(0)
        u = [x]
        for dim in dims[1:]:
            u.append(torch.zeros(batch_size, dim, requires_grad=True).to(device))
        return u
    
    @staticmethod
    @abstractmethod
    def initialize_z(x, u, nonlinearity, dims):
        """For each hidden layer and the output layer initialize the activations.
        Arguments:
            x (torch.Tensor): The input to the network.
            u (list): A list of the initialized activations.
            nonlinearity (torch.nn.functional): The nonlinearity to apply.
            dims (list): The dimensions of the network.
        Returns:
            z: A list of the initialized activations.
        """
        z = [x]
        for u_layer in u[1:]:
            z.append(nonlinearity(u_layer))
        return z
