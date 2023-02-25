import wandb

import os
import torch
from torch import optim, nn, utils, Tensor
from data.base import FaceDataset, LitDataModule
from models.autoencoder import AutoencoderKL

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from pytorch_lightning.callbacks import ModelCheckpoint


def parse_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/content/DF11',
                        help='dataset path, default set for colab env')
    parser.add_argument('--checkpoint_path', type=str, default='/content/drive/MyDrive/ldm_fmface/checkpoint',
                        help='checkpoint saving path, default set for colab env')
    parser.add_argument('--checkpoint_file', type=str, default='epoch=0-step=2000.ckpt',
                        help='checkpoint file name')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')


    parser.add_argument('--embed_dim', type=int, default=3,
                        help='intermediate dimension (input dim of latent diffusion)')
    parser.add_argument('--z_channel', type=int, default=3,
                        help='output&input dim for encoder&decoder, usually same as embed_dim')
    parser.add_argument('--resolution', type=int, default=256,
                        help='input image size')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='input dim for encoder')
    parser.add_argument('--out_ch', type=int, default=3,
                        help='output dim for decoder')
    parser.add_argument('--ch', type=int, default=128,
                        help='intermediate channel num')
    parser.add_argument('--ch_mult', type=str, default='1,2,4',
                        help='number of channels of each layer in encoder/decoder')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                        help='resnet block number')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='percentage of dropout')
    args = parser.parse_args(argv)
    return args


def main(args):
    # setup data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = FaceDataset(data_dir=args.data_path, train=True)
    train_loader = utils.data.DataLoader(dataset, shuffle=True, batch_size=args.batch_size)


    # init the autoencoder
    autoencoder = AutoencoderKL(embed_dim=args.embed_dim, 
                                z_channels=args.z_channel,
                                resolution=args.resolution,
                                in_channels=args.in_channels,
                                out_ch=args.out_ch,
                                ch = args.ch,
                                ch_mult=args.ch_mult.split(','),
                                num_res_blocks=args.num_res_blocks,
                                dropout=args.dropout,
                                monitor="val/rec_loss").to(device)

    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(project='fmface_generator', name='autoencoder_init')

    # add your batch size to the wandb config
    wandb_logger.experiment.config["batch_size"] = args.batch_size

    # model checkpoint custom
    ckpt_callback = ModelCheckpoint(dirpath=args.checkpoint_path, every_n_train_steps=2000)

    # pass wandb_logger to the Trainer 
    trainer = pl.Trainer(logger=wandb_logger,callbacks=[ckpt_callback], benchmark= True, accumulate_grad_batches=4, 
                         accelerator="gpu" if device=='cuda' else 'cpu', devices=1, 
                         resume_from_checkpoint='/content/drive/MyDrive/ldm_fmface/checkpoint/'+args.checkpoint_file)
    # train the model
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()

if __name__ == "__main__":
    arg = parse_args()
    main(arg)