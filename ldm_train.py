import wandb
import os, sys
import torch
from torch import optim, nn, utils, Tensor
from data.base import FaceDataset
from models.ldm.ddpm import LatentDiffusion

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from pytorch_lightning.callbacks import ModelCheckpoint

from config.default_config import AE_DEFAULT_CONFIG, UNET_DEFAULT_CONFIG, LDM_DEFAULT_CONFIG, COND_CONFIG

def parse_args(argv=None, **kwargs):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/content/DF11',
                        help='dataset path, default set for colab env')
    parser.add_argument('--checkpoint_path', type=str, default='/content/drive/MyDrive/ldm_fmface/checkpoint',
                        help='checkpoint saving path, default set for colab env')
    parser.add_argument('--checkpoint_file', type=str, default='',
                        help='checkpoint file name')
    parser.add_argument('--batch_size', type=int, default=3,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='epochs')
    parser.add_argument('--resume_train', action='store_true',
                        help='resume from checkpoint')
    
    for k, v in kwargs.items():
        parser.add_argument('--' + k, default=v)
    args = parser.parse_args(argv)
    return args

def main(args):
    # setup data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = FaceDataset(data_dir=args.data_path, train=True)
    train_loader = utils.data.DataLoader(dataset, shuffle=True)

    # init the autoencoder
    autoencoder = LatentDiffusion(
            linear_start= args.Linear_start, Linear_end= args.Linear_end, log_every_t= args.Log_every_t,
            timesteps= args.timesteps, first_stage_key= args.first_stage_key, image_size= args.image_size,
            channels= args.channels, monitor= args.monitor, unet_config = UNET_DEFAULT_CONFIG,
            first_stage_config = AE_DEFAULT_CONFIG, cond_stage_config = COND_CONFIG,
                                ).to(device)

    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(project='fmface_generator')

    # add your batch size to the wandb config
    wandb_logger.experiment.config["batch_size"] = args.batch_size

    # model checkpoint custom
    ckpt_callback = ModelCheckpoint(dirpath=args.checkpoint_path, every_n_train_steps=2000)

    # pass wandb_logger to the Trainer 
    trainer = pl.Trainer(logger=wandb_logger,callbacks=[ckpt_callback], benchmark= True, accumulate_grad_batches=15, accelerator="gpu" if device=='cuda' else 'cpu', devices=1)

    # train the model
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()

if __name__ == "__main__":
    #DEFAULT = dict(AE_DEFAULT_CONFIG.items()+UNET_DEFAULT_CONFIG.items()+LDM_DEFAULT_CONFIG.items())
    arg = parse_args(LDM_DEFAULT_CONFIG)
    main(arg)