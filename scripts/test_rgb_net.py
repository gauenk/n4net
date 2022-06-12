
# -- misc --
import os,math,tqdm
import pprint
pp = pprint.PrettyPrinter(indent=4)

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- optical flow --
import svnlb

# -- caching results --
import cache_io

# -- network --
import n4net

# -- lightning module --
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def run_exp(cfg):

    # -- data --
    data,loaders = data_hub.sets.load(cfg)
    index = data.te.groups.index(cfg.vid_name)
    sample = data.te[index]

    # -- unpack --
    noisy,clean = sample['noisy'],sample['clean']
    noisy,clean = noisy.to(cfg.device),clean.to(cfg.device)

    # -- network --
    model = n4net.batched_lidia.load_model(cfg.sigma).to(cfg.device)
    model.eval()
    # -- size --
    nframes = noisy.shape[0]
    ngroups = int(25 * 37./nframes)
    batch_size = ngroups*1024

    # -- optical flow --
    noisy_np = noisy.cpu().numpy()
    if cfg.comp_flow == "true":
        flows = svnlb.compute_flow(noisy_np,cfg.sigma)
    else:
        flows = None

    # -- internal adaptation --
    if cfg.internal_adapt_nsteps > 0:
        model.run_internal_adapt(noisy,cfg.sigma,flows=flows,
                                 batch_size=batch_size,
                                 nsteps=cfg.internal_adapt_nsteps,
                                 nepochs=cfg.internal_adapt_nepochs)
    # -- denoise --
    deno = model(noisy,cfg.sigma,flows=flows,batch_size=batch_size)

    # -- save example --
    out_dir = Path(cfg.saved_dir) / str(cfg.uuid)
    deno_fns = n4net.utils.io.save_burst(deno,out_dir,"deno")

    # -- psnr --
    t = clean.shape[0]
    deno = deno.detach()
    clean_rs = clean.reshape((t,-1))/255.
    deno_rs = deno.reshape((t,-1))/255.
    mse = th.mean((clean_rs - deno_rs)**2,1)
    psnrs = -10. * th.log10(mse).detach()
    psnrs = list(psnrs.cpu().numpy())

    # -- init results --
    results = edict()
    results.psnrs = psnrs
    results.deno_fn = deno_fns
    results.vid_name = cfg.vid_name

    return results


def default_cfg():
    # -- config --
    cfg = edict()
    cfg.nframes_tr = 5
    cfg.nframes_val = 5
    cfg.nframes_te = 0
    cfg.saved_dir = "./output/saved_results/"
    cfg.checkpoint_dir = "/home/gauenk/Documents/packages/n4net/output/checkpoints/"
    cfg.isize = None
    cfg.num_workers = 4
    cfg.device = "cuda:0"
    return cfg

def main():

    # cache_io
    # get_exp_grids(sigmas,internal_adapt_steps)

    # -- (0) start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_dir = ".cache_io"
    cache_name = "suite_iphone_final_v2" # current!
    cache = cache_io.ExpCache(cache_dir,cache_name)
    cache.clear()

    # -- get mesh --
    dnames = ["set8"]
    vid_names = ["snowboard","sunflower","tractor","hypersmooth",
                 "motorbike","park_joy","rafting","touchdown"]
    sigmas = [10,30,50]
    internal_adapt_nsteps = [0,100]
    internal_adapt_nepochs = [2]
    comp_flow = ["true"]
    exp_lists = {"dname":dnames,"vid_name":vid_names,"sigma":sigmas,
                 "internal_adapt_nsteps":internal_adapt_nsteps,
                 "internal_adapt_nepochs":internal_adapt_nepochs,
                 "comp_flow":comp_flow}
    exps = cache_io.mesh_pydicts(exp_lists) # create mesh
    pp.pprint(exps)

    # -- group with default --
    cfg = default_cfg()
    cache_io.append_configs(exps,cfg) # merge the two

    # -- run exps --
    nexps = len(exps)
    for exp_num,exp in enumerate(exps):

        # -- info --
        if verbose:
            print("-="*25+"-")
            print(f"Running experiment number {exp_num+1}/{nexps}")
            print("-="*25+"-")
            pp.pprint(exp)

        # -- logic --
        uuid = cache.get_uuid(exp) # assing ID to each Dict in Meshgrid
        results = cache.load_exp(exp) # possibly load result
        if results is None: # check if no result
            exp.uuid = uuid
            with th.no_grad():
                results = run_exp(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- load results --
    records = cache.load_flat_records(exps)
    print(records)


if __name__ == "__main__":
    main()
