
# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- this package --
import n4net

# -- lightning module --
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

class LitN4Net(pl.LightningModule):

    def __init__(self,sigma,batch_size=1,device="cuda:0"):
        super().__init__()
        self.sigma = sigma
        arch_cfg = n4net.ArchitectureOptions(True)
        pad_offs, total_pad = n4net.model.misc.calc_padding(arch_cfg)
        net = n4net.N4Net(pad_offs, arch_cfg).to(device)
        self.net = net
        self.batch_size = batch_size

    def forward(self,data,sigma):
        return self.net(data,sigma)

    def configure_optimizers(self):
        optim = th.optim.Adam(self.parameters(),lr=1e-3)
        return optim

    def training_step(self, batch, batch_idx):
        noisy,clean = batch['noisy'][0],batch['clean'][0]
        print("[tr] noisy.shape: ",noisy.shape)
        deno = self.net(noisy,self.sigma)
        loss = th.sum((clean - deno)**2)
        self.log("train_loss", loss, on_step=True, on_epoch=False,
                 batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        noisy,clean = batch['noisy'][0],batch['clean'][0]
        print("[val] noisy.shape: ",noisy.shape)
        deno = self.net(noisy,self.sigma)
        loss = th.sum((clean - deno)**2)
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 batch_size=self.batch_size)

    def test_step(self, batch, batch_nb):
        noisy,clean = batch['noisy'][0],batch['clean'][0]
        print("[te] noisy.shape: ",noisy.shape)
        deno = self.net(noisy,self.sigma)
        loss = th.sum((clean - deno)**2)
        self.log("test_loss", loss, on_step=False, on_epoch=True,
                 batch_size=self.batch_size)

def main():

    # -- config --
    cfg = edict()
    cfg.sigma = 30
    cfg.nframes = 5
    cfg.checkpoint_dir = "/home/gauenk/Documents/packages/n4net/output/checkpoints/"
    cfg.tr_isize = '96_96'
    cfg.val_isize = '96_96'
    cfg.num_workers = 4
    cfg.device = "cuda:0"

    # -- data --
    data,loaders = data_hub.sets.submillilux.load_paired(cfg)

    # -- load pytorch_lightning --
    model = LitN4Net(cfg.sigma).to(cfg.device)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss",save_top_k=3,mode="max",
                                          dirpath=cfg.checkpoint_dir,
                                          filename="toy-{epoch:02d}-{val_loss:.2f}",)
    trainer = pl.Trainer(gpus=1,precision=32,limit_train_batches=1.,
                         max_epochs=3,log_every_n_steps=1,
                         callbacks=[checkpoint_callback])
    trainer.fit(model, loaders.tr, loaders.val)
    trainer.test(model, loaders.te)

    #
    # -- test sample --
    #

    # -- load video --
    vid_cfg = data_hub.get_video_cfg("submillilux","seq1")
    vid = data_hub.load_video(vid_cfg)
    vid = th.from_numpy(vid).to(cfg.device)[:3,:,:96,:96]
    net = model.net.to(cfg.device)
    deno = net(vid,cfg.sigma)


if __name__ == "__main__":
    main()
