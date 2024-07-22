import torch
from torch.utils.data import Dataset, DataLoader
import torch
import os
from torchvision import transforms

class Run:
    def __init__(self, config):
        self.config = config
        # build model
        model_config = self.config.get("model_config")
        self.model = self.build_model(model_config)
        # build dataset
        dataset_config = self.config.get("dataset_config")
        self.dataset = dataset_config.get("dataset")
        data_dir = dataset_config.get("data_dir")
        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        # Load the dataset from the specified directory
        train_dataset = self.dataset(data_dir, train=True, transform=transform)
        test_dataset = self.dataset(data_dir, train=False, transform=transform)
        # Create DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        # activation initializer
        activations_optimizer_config = self.config.get("activations_optimizer_config", dict())
        self.activations_initializer = activations_optimizer_config.get("activations_initializer")
        self.activations_optimizer = activations_optimizer_config.get("activations_optimizer")
    
    def build_model(self, model_config):
        model_name = model_config.get("model_name")
        model_args = model_config.get("model_args", dict())
        load_existing = model_config.get("use_existing", False)
        load_path = model_config.get("load_path")
        model = model_name(**model_args)
        if load_existing:
            model.load_state_dict(torch.load(os.path.expanduser(load_path)))
        return model
    
    def train(self):
        activations_optimizer_config = self.config.get("activations_optimizer_config")
        for i, (x, y) in enumerate(self.train_loader):
            # prepare data
            x = x.view(x.size(0), -1)
            y = torch.nn.functional.one_hot(y.to(torch.int64), num_classes=self.model.dims[-1]).float()
            # forward pass free phase
            u_free, z_free, total_energy_free = self.model.forward(x, activations_optimizer_config, self.activations_initializer, self.activations_optimizer, u=None, z=None, y=None, mode="supervised")
            # forward pass clamped phase
            u_clamped, z_clamped, total_energy_clamped = self.model.forward(x, activations_optimizer_config, self.activations_initializer, self.activations_optimizer, u=u_free, z=z_free, y=y, mode="supervised")
            # backward pass
            self.update_weights(x, y, u_free, z_free, u_clamped, z_clamped)
            # validate
            self.validate()
    
    def update_weights(self, x, y, u_free, z_free, u_clamped, z_clamped):
        activations_optimizer_config = self.config.get("activations_optimizer_config", dict())
        weights_optimizer_config = self.config.get("weights_optimizer_config", dict())
        beta = activations_optimizer_config.get("beta", 0.01)
        lr_array = weights_optimizer_config.get("lr", 0.01)
        # iterate over model layers
        for i, layer in enumerate(self.model.layers):
            # compute gradients
            grad = z_clamped[i + 1].T @ z_clamped[i] - z_free[i + 1].T @ z_free[i]
            # update weights
            if isinstance(lr_array, list):
                lr = lr_array[i]
            else:
                lr = lr_array
            self.model.layers[i].weight.data += lr * (1.0 / beta) * grad
            # self.model.layers[i].bias.data -= lr * (1.0 / beta) * z_free[i + 1].mean(0) - z_clamped[i + 1].mean(0)
    
    def validate(self):
        activations_optimizer_config = self.config.get("activations_optimizer_config")
        for i, (x, y) in enumerate(self.test_loader):
            # prepare data
            x = x.view(x.size(0), -1)
            # forward pass free phase
            u_free, z_free, total_energy_free = self.model.forward(x, activations_optimizer_config, self.activations_initializer, self.activations_optimizer, u=None, z=None, y=None, mode="supervised")
            # prediction
            prediction = z_free[-1].argmax(1)
            # accuracy
            accuracy = (prediction == y).float().mean()
            print(f"Accuracy: {accuracy}")
            break
            

        





# DATA_DIR = "/teamspace/studios/this_studio/PhysicalLearning/data/MNIST/raw/"
# DATASET = MNISTDataset

# run = Run(dataset=DATASET, data_dir=DATA_DIR)

# # Define transformations
# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])

# # Load the MNIST dataset from the specified directory
# train_dataset = MNISTDataset(data_dir, train=True, transform=transform)
# test_dataset = MNISTDataset(data_dir, train=False, transform=transform)


