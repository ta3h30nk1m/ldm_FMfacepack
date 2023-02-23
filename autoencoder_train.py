import wandb

import os
import torch
from torch import optim, nn, utils, Tensor
from data.base import FaceDatasetTrain
from models.autoencoder import AutoencoderKL

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger



# setup data
batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = FaceDatasetTrain()
train_loader = utils.data.DataLoader(dataset, shuffle=True)

# init the autoencoder
autoencoder = AutoencoderKL(embed_dim=3, monitor="val/rec_loss").to(device)

# initialise the wandb logger and name your wandb project
wandb_logger = WandbLogger(project='fmface_generator')

# add your batch size to the wandb config
wandb_logger.experiment.config["batch_size"] = batch_size

# pass wandb_logger to the Trainer 
trainer = pl.Trainer(logger=wandb_logger, benchmark= True, accumulate_grad_batches=2, default_root_dir="/content/drive/MyDrive/ldm_fmface", accelerator="gpu" if device=='cuda' else 'cpu', devices=1)

# train the model
trainer.fit(model=autoencoder, train_dataloaders=train_loader)

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()