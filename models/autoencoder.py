import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager

from .ldm.modules.quantize import VectorQuantizer2 as VectorQuantizer

from .ldm.modules.diffusionmodules.model import Encoder, Decoder
from .ldm.modules.losses import LPIPSWithDiscriminator
from .ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from .ldm.modules.ema import LitEma

class VQModel(pl.LightningModule):
    def __init__(self,
                 embed_dim=3,
                 n_embed=8192,
                 z_channels=3,
                 resolution=256,
                 in_channels=3,
                 out_ch=3,
                 ch=128,
                 ch_mult=[1,2,4],
                 num_res_blocks=2,
                 dropout=0.0,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 train_disc = False,
                 batch_resize_range=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(double_z= False, z_channels=z_channels, resolution=resolution,in_channels=in_channels,out_ch=out_ch,ch=ch,ch_mult=ch_mult,
                                num_res_blocks=num_res_blocks,attn_resolutions=[ ], dropout= dropout)
        self.decoder = Decoder(double_z= False, z_channels=z_channels, resolution=resolution,in_channels=in_channels,out_ch=out_ch,ch=ch,ch_mult=ch_mult,
                                num_res_blocks=num_res_blocks,attn_resolutions=[ ], dropout= dropout)
        self.loss = LPIPSWithDiscriminator( disc_start= 50001, kl_weight= 0.000001, disc_weight= 0.5)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(out_ch, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, out_ch, 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.lr_g_factor = lr_g_factor

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_,_,ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                # do the first few batches with max size to avoid later oom
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size+16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            x = x.detach()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train",
                                            predicted_indices=ind)

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val"+suffix,
                                        predicted_indices=ind
                                        )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            predicted_indices=ind
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class VQModelInterface(VQModel):
    def __init__(self, embed_dim=3,
                 n_embed=8192,
                 z_channels=3,
                 resolution=256,
                 in_channels=3,
                 out_ch=3,
                 ch=128,
                 ch_mult=[1,2,4],
                 num_res_blocks=2,
                 dropout=0.0,
                 ckpt_path=None):
        super().__init__(embed_dim=embed_dim, n_embed=n_embed,
                 z_channels=z_channels,
                 resolution=resolution,
                 in_channels=in_channels,
                 out_ch=out_ch,
                 ch=ch,
                 ch_mult=ch_mult,
                 num_res_blocks=num_res_blocks,
                 dropout=dropout,
                 ckpt_path=ckpt_path
                 )
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 embed_dim=3,
                 z_channels=3,
                 resolution=256,
                 in_channels=3,
                 out_ch=3,
                 ch=128,
                 ch_mult=[1,2,4],
                 num_res_blocks=2,
                 dropout=0.0,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 train_disc = False,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(double_z= True, z_channels=z_channels, resolution=resolution,in_channels=in_channels,out_ch=out_ch,ch=ch,ch_mult=ch_mult,
                                num_res_blocks=num_res_blocks,attn_resolutions=[ ], dropout= dropout)
        self.decoder = Decoder(double_z= True, z_channels=z_channels, resolution=resolution,in_channels=in_channels,out_ch=out_ch,ch=ch,ch_mult=ch_mult,
                                num_res_blocks=num_res_blocks,attn_resolutions=[ ], dropout= dropout)
        self.loss = LPIPSWithDiscriminator( disc_start= 50001, kl_weight= 0.000001, disc_weight= 0.5)
        self.learning_rate = 4.5e-6
        #assert double_z == True
        self.quant_conv = torch.nn.Conv2d(2*3, 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, 3, 1)
        self.embed_dim = embed_dim
        self.train_disc = train_disc
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.ckpt = ckpt_path
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx):
        #inputs = self.get_input(batch, self.image_key)
        inputs = batch.cuda()
        reconstructions, posterior = self(inputs)

        if not self.train_disc:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, self.train_disc, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return aeloss

        elif self.train_disc:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, self.train_disc, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return discloss

    # def validation_step(self, batch, batch_idx):
    #     inputs = self.get_input(batch, self.image_key)
    #     reconstructions, posterior = self(inputs)
    #     aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
    #                                     last_layer=self.get_last_layer(), split="val")

    #     discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
    #                                         last_layer=self.get_last_layer(), split="val")

    #     self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
    #     self.log_dict(log_dict_ae)
    #     self.log_dict(log_dict_disc)
    #     return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        if not self.train_disc:
            opt = torch.optim.Adam(list(self.encoder.parameters())+
                                    list(self.decoder.parameters())+
                                    list(self.quant_conv.parameters())+
                                    list(self.post_quant_conv.parameters()),
                                    lr=lr, betas=(0.5, 0.9))
            if self.ckpt is not None:
                opt_state_dict = torch.load(self.ckpt, map_location='cpu')['optimizer_states'][0]
                opt.load_state_dict(opt_state_dict)
        elif self.train_disc:
            opt = torch.optim.Adam(self.loss.discriminator.parameters(),
                                        lr=lr, betas=(0.5, 0.9))
            if self.ckpt is not None:
                opt_state_dict = torch.load(self.ckpt, map_location='cpu')['optimizer_states']
                opt.load_state_dict(opt_state_dict)
        return opt
        #return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x