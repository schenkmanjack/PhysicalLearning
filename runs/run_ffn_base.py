import torch
from torch.utils.data import Dataset, DataLoader
import torch
import os
from torchvision import transforms

class RunFFNBase:
    """Class for Run of a feed forward network trained with basic backpropagation."""
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dim = self.config.get("output_dim", 10)
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
        # loss function
        self.loss_func = self.config.get("loss_func", torch.nn.MSELoss())
        # optimizer
        optimizer_config = self.config.get("weights_optimizer_config")
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=optimizer_config.get("lr"))
    
    def build_model(self, model_config):
        model_name = model_config.get("model_name")
        model_args = model_config.get("model_args", dict())
        load_existing = model_config.get("use_existing", False)
        load_path = model_config.get("load_path")
        model = model_name(**model_args)
        if load_existing:
            model.load_state_dict(torch.load(os.path.expanduser(load_path)))
        return model
    
    def compute_loss(self, prediction, y):
        return self.loss_func(prediction, y)

    def train(self):
        activations_optimizer_config = self.config.get("activations_optimizer_config")
        for epoch in range(self.config.get("num_epochs", 10)):
            self.model.train()
            for i, (x, y) in enumerate(self.train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                # prepare data
                x = self.prepare_data(x)
                # batch norm
                # x = (x - x.mean(0)) / x.std(0)
                y = torch.nn.functional.one_hot(y.to(torch.int64), num_classes=self.output_dim).float()
                # forward pass 
                output = self.model.forward(x)
                # loss
                prediction = output
                loss = self.compute_loss(prediction, y)
                # backward pass
                self.update_weights(loss)
                # if i > 1:
                #     break
            # validate
            accuracy = self.validate()
            print(accuracy)
            self.extra_validation()

    def prepare_data(self, x):
        return x.view(x.size(0), -1)

    
    def update_weights(self, loss):
        # get grads
        self.model.zero_grad()
        loss.backward()
        # update weights
        self.optimizer.step()

    
    def validate(self):
        activations_optimizer_config = self.config.get("activations_optimizer_config")
        self.model.eval()
        for i, (x, y) in enumerate(self.test_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            # prepare data
            x = self.prepare_data(x)
            # forward pass free phase
            output = self.model.forward(x)
            # prediction
            prediction = output
            prediction = prediction.argmax(1)
            # accuracy
            accuracy = (prediction == y).float().mean()
            return accuracy
    
    def extra_validation(self):
        pass
            

        
