import torch
from torchvision import datasets, transforms

# Specify the directory to save the dataset
save_dir = "/teamspace/studios/this_studio/PhysicalLearning/data"

# Download and save the MNIST dataset
datasets.MNIST(save_dir, train=True, download=True, transform=transforms.ToTensor())
