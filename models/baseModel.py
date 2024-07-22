import torch
import torch.nn as nn

class baseModel(nn.Module):
    def __init__(self, dims, nonlinearity):
        self.dims = dims
        self.nonlinearity = nonlinearity
        self.layers = []
        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i + 1], bias=True)
            layer.bias.data.fill_(0.0)
            self.layers.append(layer)

    
    def forward(self, x, optimizer_config, activations_initializer, activations_optimizer, u=None, z=None, y=None, mode="supervised"):
        # initialize u and z
        if u is None or z is None:
            u = activations_initializer.initialize_u(x, dims=self.dims)
            z = activations_initializer.initialize_z(x, u, self.nonlinearity, self.dims)
        # optimize activations
        activations_optimizer = activations_optimizer(optimizer_config)
        u, z, total_energy = activations_optimizer.optimize(x, u, z, y=y, model=self, mode=mode)
        return u, z, total_energy
    


    

            