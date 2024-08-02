import torch
from datasets import MNISTDataset
from runs import RunFFNBase
from helpers import ConnectionCuttingFFN
from models import VisionTransformerModel

# define the run
class ViTBaseRun(RunFFNBase):
    def prepare_data(self, x):
        return x.permute(0, 2, 1, 3)

    
# dataset
DATA_DIR = "/teamspace/studios/this_studio/PhysicalLearning/data/MNIST/raw/"
DATASET = MNISTDataset
BATCH_SIZE_TRAIN = 512
BATCH_SIZE_TEST = 256
# model
MODEL_NAME = VisionTransformerModel
N_CHANNELS = 1
EMBED_DIM = 64
N_LAYERS = 6
N_ATTENTION_HEADS = 4
FORWARD_MUL = 2
IMAGE_SIZE = 28
PATCH_SIZE = 7
N_CLASSES = 10
DROPOUT = 0.1

# weight optimizer
LR =  1e-2
NUM_EPOCHS = 1500
LOSS_FUNC = torch.nn.CrossEntropyLoss()


# create the confiuration 
CONFIG = dict(
    batch_size_train=BATCH_SIZE_TRAIN,
    batch_size_test=BATCH_SIZE_TEST,
    output_dim=N_CLASSES,
    num_epochs=NUM_EPOCHS,
    loss_func=LOSS_FUNC,
    dataset_config=dict(
        dataset=MNISTDataset,
        data_dir=DATA_DIR,
    ),
    model_config=dict(
        model_name=MODEL_NAME,
        model_args=dict(
        n_channels=N_CHANNELS, 
        embed_dim=EMBED_DIM, 
        n_layers=N_LAYERS, 
        n_attention_heads=N_ATTENTION_HEADS, 
        forward_mul=FORWARD_MUL, 
        image_size=IMAGE_SIZE, 
        patch_size=PATCH_SIZE, 
        n_classes=N_CLASSES, 
        dropout=DROPOUT
        ),
    ),
    weights_optimizer_config=dict(
        lr=LR,
    ),
)

run = ViTBaseRun(config=CONFIG)
run.train()


