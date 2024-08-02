import torch

class ConnectionCuttingEBM:
    
    def get_mask(self, x):
        connection_cutting_config = self.config.get("connection_cutting_config", dict())
        fraction = connection_cutting_config.get("fraction", 0.5)
        mask_during_training = connection_cutting_config.get("mask_during_training", False)
        model = self.model
        if mask_during_training:
            mask = []
            for layer in model.layers[:-1]:
                num_neurons = layer.weight.data.size(0)
                mask_layer = torch.rand(x.shape[0], num_neurons).to(self.device) < fraction
                mask.append(mask_layer)
            return mask
        super(ConnectionCuttingEBM, self).get_mask(x)

    def extra_validation(self):
        """Randomly set to zero fraction of model weights."""
        connection_cutting_config = self.config.get("connection_cutting_config", dict())
        fraction = connection_cutting_config.get("fraction", 0.5)
        model = self.model
        mask_weights = connection_cutting_config.get("mask_weights", False)
        mask = None
        if mask_weights:
            model = self.build_model(self.config.get("model_config"))
            model.load_state_dict(self.model.state_dict())
            model = model.to(self.device)
            for layer in model.layers:
                mask = torch.rand_like(layer.weight.data).to(self.device) < fraction
                layer.weight.data = layer.weight.data * mask
        else:
            mask = []
            for layer in model.layers[:-1]:
                num_neurons = layer.weight.data.size(0)
                mask_layer = torch.rand(self.batch_size_test, num_neurons).to(self.device) < fraction
                mask.append(mask_layer)
                
        # validate
        activations_optimizer_config = self.config.get("activations_optimizer_config")
        validation_config = activations_optimizer_config.get("validation_config", dict())
        epsilon = validation_config.get("epsilon", 0.1)
        num_iterations = validation_config.get("num_iterations", 20)
        for i, (x, y) in enumerate(self.test_loader):
            # prepare data
            x = x.to(self.device)
            y = y.to(self.device)
            x = x.view(x.size(0), -1)
            # forward pass free phase
            u_free, z_free, total_energy_free = model.forward(x, activations_optimizer_config, self.activations_initializer, self.activations_optimizer, u=None, z=None, y=None, num_iterations=num_iterations, mask=mask, mode="supervised")
            # prediction
            prediction = z_free[-1].argmax(1)
            # accuracy
            accuracy = (prediction == y).float().mean()
            print("connection cutting accuracy: ", accuracy)
            return