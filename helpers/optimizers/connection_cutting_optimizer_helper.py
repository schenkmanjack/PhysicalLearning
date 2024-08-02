import torch

class ConnectionCuttingOptimizerHelper:
    def per_iteration(self, model, run):
        model = super().per_iteration(model, run)
        if run.model.training:
            return model
        model = run.build_model(run.config.get("model_config"))
        # get state dict from self.model and load into model
        connection_cutting_config = run.config.get("connection_cutting_config", dict())
        fraction = connection_cutting_config.get("fraction", 0.5)
        model.load_state_dict(run.model.state_dict())
        model = model.to(run.device)
        for layer in model.layers:
            mask = torch.rand_like(layer.weight.data) < fraction
            layer.weight.data = layer.weight.data * mask
        return model