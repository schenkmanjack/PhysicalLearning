import torch
from torch.utils.data import Dataset, DataLoader
import torch
import os
from torchvision import transforms
from .run_ebm_base import RunEBMBase

class RunEBMSymmetricUpdate(RunEBMBase):
    """This is a Run class for energy based models trained using equilibrium propogation with symmetric updates."""
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
                # forward pass positive clamped phase
                num_iterations_clamped = activations_optimizer_config.get("num_iterations_clamped", 4)
                u_clamped_pos, z_clamped_pos, total_energy_clamped_pos = self.model.forward(x, activations_optimizer_config, self.activations_initializer, self.activations_optimizer, u=u_free, z=z_free, y=y, mode="supervised", num_iterations=num_iterations_clamped, mask=mask, run=self)
                # forward pass negative clamped phase
                num_iterations_clamped = activations_optimizer_config.get("num_iterations_clamped", 4)
                beta = activations_optimizer_config.get("beta", 0.01)
                u_clamped_neg, z_clamped_neg, total_energy_clamped_neg = self.model.forward(x, activations_optimizer_config, self.activations_initializer, self.activations_optimizer, u=u_free, z=z_free, y=y, beta = -1 * beta, mode="supervised", num_iterations=num_iterations_clamped, mask=mask, run=self)
                # backward pass
                self.update_weights(x, y, u_free=u_clamped_neg, z_free=z_clamped_neg, u_clamped=u_clamped_pos, z_clamped=z_clamped_pos)
                # if i > 1:
                #     break
            # validate
            accuracy = self.validate()
            print(accuracy)
            self.extra_validation()

    def validate(self):
        activations_optimizer_config = self.config.get("activations_optimizer_config")
        validation_config = activations_optimizer_config.get("validation_config", dict())
        self.model.eval()
        for i, (x, y) in enumerate(self.test_loader):
            # prepare data
            x = x.to(self.device)
            x = x.view(x.size(0), -1)
            y = y.to(self.device)
            #get mask
            mask = None
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


