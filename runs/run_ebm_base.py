import torch
from torch.utils.data import Dataset, DataLoader
import torch
import os
from torchvision import transforms

class RunEBMBase:
    """This is the basic energy based model Run class. The code is written for a feed forward network trained using
    equilibrium propogation."""
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # build model
        model_config = self.config.get("model_config")
        self.model = self.build_model(model_config)
        self.model.to(self.device)
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
        self.batch_size_train = self.config.get("batch_size_train", 512)
        self.batch_size_test = self.config.get("batch_size_test", 256)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size_train, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size_test, shuffle=False)
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
        for epoch in range(self.config.get("num_epochs", 10)):
            self.model.train()
            for i, (x, y) in enumerate(self.train_loader):
                # prepare data
                x = x.to(self.device)
                x = x.view(x.size(0), -1)
                # batch norm
                # x = (x - x.mean(0)) / x.std(0)
                y = y.to(self.device)
                y = torch.nn.functional.one_hot(y.to(torch.int64), num_classes=self.model.dims[-1]).float()
                #get mask
                mask = self.get_mask(x)
                # forward pass free phase
                num_iterations_free = activations_optimizer_config.get("num_iterations_free", 20)
                u_free, z_free, total_energy_free = self.model.forward(x, activations_optimizer_config, self.activations_initializer, self.activations_optimizer, u=None, z=None, y=None, mode="supervised", num_iterations=num_iterations_free, mask=mask, run=self)
                # forward pass clamped phase
                num_iterations_clamped = activations_optimizer_config.get("num_iterations_clamped", 4)
                u_clamped, z_clamped, total_energy_clamped = self.model.forward(x, activations_optimizer_config, self.activations_initializer, self.activations_optimizer, u=u_free, z=z_free, y=y, mode="supervised", num_iterations=num_iterations_clamped, mask=mask, run=self)
                # backward pass
                self.update_weights(x, y, u_free, z_free, u_clamped, z_clamped)
                # if i > 1:
                #     break
            # validate
            accuracy = self.validate()
            print(accuracy)
            self.extra_validation()

    
    def update_weights(self, x, y, u_free, z_free, u_clamped, z_clamped):
        activations_optimizer_config = self.config.get("activations_optimizer_config", dict())
        weights_optimizer_config = self.config.get("weights_optimizer_config", dict())
        beta = activations_optimizer_config.get("beta", 0.01)
        lr_array = weights_optimizer_config.get("lr", 0.01)
        # iterate over model layers
        for i, layer in enumerate(self.model.layers):
            # compute gradients
            grad = z_clamped[i + 1].T @ z_clamped[i] - z_free[i + 1].T @ z_free[i]
            grad = grad / x.size(0)
            # update weights
            if isinstance(lr_array, list):
                lr = lr_array[i]
            else:
                lr = lr_array
            self.model.layers[i].weight.data += lr * (1.0 / beta) *  grad #100 * torch.randn_like(grad).to(grad.device)
            # self.model.layers[i].bias.data -= 0.0001 * lr * (1.0 / beta) * z_free[i + 1].mean(0) - z_clamped[i + 1].mean(0)
    
    def validate(self):
        activations_optimizer_config = self.config.get("activations_optimizer_config")
        validation_config = activations_optimizer_config.get("validation_config", dict())
        self.model.eval()
        for i, (x, y) in enumerate(self.test_loader):
            # prepare data
            x = x.to(self.device)
            x = x.view(x.size(0), -1)
            y = y.to(self.device)
            # get mask
            mask = self.get_mask(x)
            # forward pass free phase
            epsilon = validation_config.get("epsilon", 0.1)
            num_iterations = validation_config.get("num_iterations", 20)
            print(epsilon, num_iterations)
            u_free, z_free, total_energy_free = self.model.forward(x, activations_optimizer_config, self.activations_initializer, self.activations_optimizer, u=None, z=None, y=None, epsilon=epsilon, num_iterations=num_iterations, mode="supervised", mask=mask, run=self)
            # prediction
            prediction = z_free[-1].argmax(1)
            # accuracy
            accuracy = (prediction == y).float().mean()
            return accuracy
    
    def extra_validation(self):
        pass

    def get_mask(self, x):
        mask = None
        return mask
            

        





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


