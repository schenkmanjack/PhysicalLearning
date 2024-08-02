import torch
from datasets import MNISTDataset
from runs import RunFFNBase
from helpers import ConnectionCuttingFFN
from models import baseFFNModel

# define the run
class FFNBaseRun(ConnectionCuttingFFN, RunFFNBase):
    pass

    
# dataset
DATA_DIR = "/teamspace/studios/this_studio/PhysicalLearning/data/MNIST/raw/"
DATASET = MNISTDataset
# model
MODEL_NAME = baseFFNModel
USE_BIAS = False
MODEL_DIMS = [784, 500, 10]
MODEL_NONLINEARITY = torch.nn.functional.hardsigmoid


# weight optimizer
LR =  1e-2
NUM_EPOCHS = 1200
# connection cutting
CONNECTION_CUTTING_FRACTION = 0.05

# create the confiuration 
CONFIG = dict(
    num_epochs=NUM_EPOCHS,
    dataset_config=dict(
        dataset=MNISTDataset,
        data_dir=DATA_DIR,
    ),
    model_config=dict(
        model_name=MODEL_NAME,
        model_args=dict(
        dims=MODEL_DIMS,
        nonlinearity=MODEL_NONLINEARITY,
        use_bias=USE_BIAS,
        dropout=True,
        p_dropout=CONNECTION_CUTTING_FRACTION,
        ),
    ),
    weights_optimizer_config=dict(
        lr=LR,
    ),
)

run = FFNBaseRun(config=CONFIG)
run.train()


