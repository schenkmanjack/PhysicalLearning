import torch
import torch.nn as nn

class baseModel(nn.Module):
    def __init__(self, dims, nonlinearity, use_bias=True):
        # super(baseModel, self).__init__()
        super().__init__()
        self.use_bias = use_bias
        self.dims = dims
        self.nonlinearity = nonlinearity
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i + 1], bias=use_bias)
            layer.bias.data.fill_(0.0)
            self.layers.append(layer)

    
    def forward(self, x, optimizer_config, activations_initializer, activations_optimizer, u=None, z=None, y=None, beta=None, mode="supervised", num_iterations=None, epsilon=None, mask=None, run=None):
        # initialize u and z
        if u is None or z is None:
            u = activations_initializer.initialize_u(x, dims=self.dims)
            z = activations_initializer.initialize_z(x, u, self.nonlinearity, self.dims)
        # optimize activations
        activations_optimizer = activations_optimizer(optimizer_config)
        u_new, z_new, total_energy = activations_optimizer.optimize(x, u, z, y=y, model=self, mode=mode, num_iterations=num_iterations, beta=beta, epsilon=epsilon, mask=mask, run=run)
        return u_new, z_new, total_energy
    


    

            