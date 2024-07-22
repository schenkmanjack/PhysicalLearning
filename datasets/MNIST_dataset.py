import os
import gzip
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MNISTDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        if self.train:
            self.data_path = os.path.join(root, 'train-images-idx3-ubyte.gz')
            self.labels_path = os.path.join(root, 'train-labels-idx1-ubyte.gz')
        else:
            self.data_path = os.path.join(root, 't10k-images-idx3-ubyte.gz')
            self.labels_path = os.path.join(root, 't10k-labels-idx1-ubyte.gz')

        self.data = self.load_images(self.data_path)
        self.labels = self.load_labels(self.labels_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def load_images(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            data = data.reshape(-1, 1, 28, 28)  # reshape to (N, 1, 28, 28)
        return data

    def load_labels(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

# Specify the directory where the dataset is saved
# data_dir = "/teamspace/studios/this_studio/PhysicalLearning/data/MNIST/raw/"

# # Define transformations
# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])

# # Load the MNIST dataset from the specified directory
# train_dataset = MNISTDataset(data_dir, train=True, transform=transform)
# test_dataset = MNISTDataset(data_dir, train=False, transform=transform)

# # Create DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# # Example usage: iterate through the training data
# for batch_idx, (data, target) in enumerate(train_loader):
#     print(f'Batch {batch_idx}: data shape {data.shape}, target shape {target.shape}')
#     print(type(data))
#     # Add your training code here
#     break  # Remove this line to iterate through the entire dataset
