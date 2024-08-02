import torch

class ConnectionCuttingFFN:
    def extra_validation(self):
        """Randomly set to zero fraction of model weights."""

        connection_cutting_config = self.config.get("connection_cutting_config", dict())
        fraction = connection_cutting_config.get("fraction", 0.5)
        model = self.model
        mask_weights = connection_cutting_config.get("mask_weights", False)
        mask = None
        if mask_weights:
            model = self.build_model(self.config.get("model_config"))
            # get state dict from self.model and load into model
            connection_cutting_config = self.config.get("connection_cutting_config", dict())
            fraction = connection_cutting_config.get("fraction", 0.5)
            model.load_state_dict(self.model.state_dict())
            model = model.to(self.device)
            for layer in model.layers:
                mask = torch.rand_like(layer.weight.data) < fraction
                layer.weight.data = layer.weight.data * mask

        # validate
        model.train()
        activations_optimizer_config = self.config.get("activations_optimizer_config")
        for i, (x, y) in enumerate(self.test_loader):
            # prepare data
            x = x.to(self.device)
            y = y.to(self.device)
            x = x.view(x.size(0), -1)
            # forward pass free phase
            output = model.forward(x)
            # prediction
            prediction = output.argmax(1)
            # accuracy
            accuracy = (prediction == y).float().mean()
            print("connection cutting accuracy: ", accuracy)
            return