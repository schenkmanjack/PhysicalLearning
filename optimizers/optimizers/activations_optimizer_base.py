import torch

class ActivationsOptimizerBase:
    def __init__(self, config):
        self.config = config
    
    def optimize(self, x, u_arg, z_arg, y, model, mode, beta=None, epsilon=None, num_iterations=None, mask=None, run=None):
        u = [u_layer.detach().clone() for u_layer in u_arg]
        z = [z_layer.detach().clone() for z_layer in z_arg]
        layers = model.layers
        if num_iterations is None:
            num_iterations = self.config.get("num_iterations", 20)
        if epsilon is None:
            epsilon = self.config.get("epsilon", 0.5)
        if beta is None:
            beta = self.config.get("beta", 0.01)
        
        # to have gradients
        for j in range(len(layers)):
            u[j + 1].requires_grad = True
            z[j + 1] = model.nonlinearity(u[j + 1])
        # z[-1] = u[-1]

        for i in range(num_iterations):
            model = self.per_iteration(model, run)
            layers = model.layers
            # get energy
            total_energy, energy, c = self.energy(x, u, z, y, layers, mode, beta)
            # get gradients
            grad_array = self.gradient(x, u, z, y, layers, mode, beta)
            # update activations
            num_layers = len(layers) - 1
            if mode == "supervised" and y is not None:
                num_layers += 1
            with torch.no_grad():
                for j in range(len(layers)):
                    u[j + 1] = u[j + 1] + epsilon * grad_array[j]
                    if mask is not None and j < len(layers) - 1:
                        u[j + 1] = u[j + 1] * mask[j]
                    # u[j+1] = u[j+1].clamp(min=0, max=1)
                # if y is None:
                #     u[-1] = layers[-1](model.nonlinearity(u[-2]))
                # z[-1] = u[-1]
            # update activations post-nonlinearity
            for j in range(len(layers)):
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
            # interaction_energy += torch.sum(z[i + 1] * layer.bias) # not for input which is clamped
        energy = 0.5 * u_energy - interaction_energy
        # cost
        total_energy = energy.detach().clone()
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
        for i, layer in enumerate(layers):
            z[i + 1].backward(torch.ones_like(z[i + 1]), retain_graph=False)
        
        for i, layer in enumerate(layers_relevant):
            # grad_internal = u[i + 1].grad * u[i + 1] - u[i + 1]
            grad_internal = u[i + 1].grad * (z[i] @ layers[i].weight.data.T +  z[i + 2] @ layers[i + 1].weight.data + layers[i].bias.data) - u[i + 1]
            grad_array.append(grad_internal)
        # external energy gradient
        grad_output = u[-1].grad * (z[-2] @ layers[-1].weight.data.T + layers[-1].bias.data) - z[-1]
        # external forcing gradient (if applicable)
        if mode == "supervised" and y is not None:
            grad_output  += beta * (y - z[-1])
        grad_array.append(grad_output)
        return grad_array
    
    def per_iteration(self, model, run):
        return model

        