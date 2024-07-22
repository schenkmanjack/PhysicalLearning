import torch
from datasets import MNISTDataset
from runs import Run
from models import baseModel
from optimizers import ActivationsInitializerBase, ActivationsOptimizerBase

# dataset
DATA_DIR = "/teamspace/studios/this_studio/PhysicalLearning/data/MNIST/raw/"
DATASET = MNISTDataset
# model
MODEL_NAME = baseModel
MODEL_DIMS = [784, 128, 10]
MODEL_NONLINEARITY = torch.nn.functional.hardsigmoid
# activation optimizer
ACTIVATIONS_INITIALIZER = ActivationsInitializerBase
ACTIVATIONS_OPTIMIZER = ActivationsOptimizerBase
BETA = 1.0
# weight optimizer
LR = [0.01, 0.01]
# create the confiuration 
CONFIG = dict(
    dataset_config=dict(
        dataset=MNISTDataset,
        data_dir=DATA_DIR,
    ),
    model_config=dict(
        model_name=MODEL_NAME,
        model_args=dict(
        dims=MODEL_DIMS,
        nonlinearity=MODEL_NONLINEARITY,
        ),
    ),
    activations_optimizer_config=dict(
        activations_initializer=ACTIVATIONS_INITIALIZER,
        activations_optimizer=ACTIVATIONS_OPTIMIZER,
        beta=BETA,
    ),
    weights_optimizer_config=dict(
        lr=LR,
    ),
)

run = Run(config=CONFIG)
run.train()


