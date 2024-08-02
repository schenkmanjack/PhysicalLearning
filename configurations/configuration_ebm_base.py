import torch
from datasets import MNISTDataset
from runs import RunEBMSymmetricUpdate
from helpers import ConnectionCuttingEBM, ConnectionCuttingOptimizerHelper
from models import baseModel
from optimizers import ActivationsInitializerBase, ActivationsOptimizerBase

# define the run
class RobustnessBase(ConnectionCuttingEBM, RunEBMSymmetricUpdate):
    pass

# define the optimizer
class ActivationsOptimizer(ActivationsOptimizerBase):
    pass
    
# dataset
DATA_DIR = "/teamspace/studios/this_studio/PhysicalLearning/data/MNIST/raw/"
DATASET = MNISTDataset
# model
MODEL_NAME = baseModel
USE_BIAS = True
MODEL_DIMS = [784, 500, 10]
MODEL_NONLINEARITY = torch.nn.functional.hardsigmoid
# activation optimizer
ACTIVATIONS_INITIALIZER = ActivationsInitializerBase
ACTIVATIONS_OPTIMIZER = ActivationsOptimizer
BETA = 0.2
EPSILON = 0.5
EPSILON_VALIDATE = 0.5
NUM_ITERATIONS_VALIDATE = 1500
NUM_ITERATIONS_FREE = 100 #25
NUM_ITERATIONS_CLAMPED = 30 #10
# weight optimizer
LR =  [0.1, 0.05] #[0.1, 0.05]
NUM_EPOCHS = 1200
BATCH_SIZE_TRAIN = 512
BATCH_SIZE_TEST = 256
# connection cutting
CONNECTION_CUTTING_FRACTION = 0.5
MASK_DURING_TRAINING = False

# create the confiuration 
CONFIG = dict(
    num_epochs=NUM_EPOCHS,
    batch_size_train=BATCH_SIZE_TRAIN,
    batch_size_test=BATCH_SIZE_TEST,
    dataset_config=dict(
        dataset=MNISTDataset,
        data_dir=DATA_DIR,
    ),
    model_config=dict(
        model_name=MODEL_NAME,
        model_args=dict(
        dims=MODEL_DIMS,
        nonlinearity=MODEL_NONLINEARITY,
        use_bias=USE_BIAS
        ),
    ),
    activations_optimizer_config=dict(
        activations_initializer=ACTIVATIONS_INITIALIZER,
        activations_optimizer=ACTIVATIONS_OPTIMIZER,
        beta=BETA,
        epsilon=EPSILON,
        num_iterations_free=NUM_ITERATIONS_FREE,
        num_iterations_clamped=NUM_ITERATIONS_CLAMPED,
        validation_config=dict(
            epsilon=EPSILON_VALIDATE,
            num_iterations=NUM_ITERATIONS_VALIDATE,
        ),
    ),
    weights_optimizer_config=dict(
        lr=LR,
    ),
    connection_cutting_config=dict(
        fraction=CONNECTION_CUTTING_FRACTION,
        mask_during_training=MASK_DURING_TRAINING,
    ),
)

run = RobustnessBase(config=CONFIG)
run.train()


