import torch

class ActivationsOptimizerBase:
    def __init__(self, config):
        self.config = config
    
    def optimize(self, x, u, z, y, model, mode, beta=None):
        layers = model.layers
        num_iterations = self.config.get("num_iterations", 100)
        epsilon = self.config.get("epsilon", 0.5)
        if beta is None:
            beta = self.config.get("beta", 0.01)
        for i in range(num_iterations):
            # get energy
            total_energy, energy, c = self.energy(x, u, z, y, layers, mode, beta)
            # get gradients
            grad_array = self.gradient(x, u, z, y, layers, mode, beta)
            # update activations
            num_layers = len(layers) - 1
            if mode == "supervised" and y is not None:
                num_layers += 1
            with torch.no_grad():
                for j in range(num_layers):
                    u[j + 1] = u[j + 1] + epsilon * grad_array[j]
                    u[j + 1] = u[j + 1].clamp(min=0, max=1)
                if y is None:
                    u[-1] = layers[-1](model.nonlinearity(u[-2]))
                    z[-1] = model.nonlinearity(u[-1])
            # update activations post-nonlinearity
            for j in range(num_layers):
                u[j + 1].requires_grad = True
                z[j + 1] = model.nonlinearity(u[j + 1])
        return u, z, total_energy
            

    def energy(self, x, u, z, y, layers, mode, beta):
        # compute u_energy
        u_energy = 0
        for u_layer in u:
            u_energy += torch.square(u_layer).sum()
        # compute interaction energy
        interaction_energy = 0
        # ignore final layer if the
        layers_relevant = layers
        if mode == "supervised":
            layers_relevant = layers_relevant[:-1]
        for i, layer in enumerate(layers_relevant): # axis of summation be careful
            pairwise_energy = (z[i] @ layers[i].weight.data.T + z[i + 2] @ layers[i + 1].weight.data) * z[i + 1]  # maybe problem since output is a y?
            interaction_energy += 0.5 * torch.sum(pairwise_energy)
            interaction_energy += torch.sum(z[i + 1] * layer.bias) # not for input which is clamped
        energy = 0.5 * u_energy - interaction_energy
        # cost
        total_energy = energy
        c = None
        if y is not None: # need z[-1] to be changed
            c = beta * 0.5 * torch.sum(torch.square(y - z[-1]))
            total_energy += c
        return total_energy, energy, c
    
    def gradient(self, x, u, z, y, layers, mode, beta):
        # internal energy gradient
         # ignore final layer if the
        layers_relevant = layers
        if mode == "supervised":
            layers_relevant = layers_relevant[:-1] 
        grad_array= []
        for i, layer in enumerate(layers_relevant):
            z[i + 1].backward(torch.ones_like(z[i + 1]), retain_graph=True)
        
        for i, layer in enumerate(layers_relevant):
            # grad_internal = u[i + 1].grad * u[i + 1] - u[i + 1]
            grad_internal = u[i + 1].grad * (layers[i](z[i]) +  z[i + 2] @ layers[i + 1].weight.data) - u[i + 1]
            grad_array.append(grad_internal)

        # external forcing gradient (if applicable)
        if mode == "supervised" and y is not None:
            grad_external = beta * (z[-1] - y)
            grad_array.append(grad_external)
        return grad_array

        