"""

When first created our model is identical to lidia

"""

# -- misc --
import sys,tqdm,pytest
from pathlib import Path

# -- dict data --
import copy
from easydict import EasyDict as edict

# -- vision --
from PIL import Image

# -- testing --
import unittest
import tempfile

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- data --
import data_hub

# -- package imports [to test] --
import dnls # supporting
import n4net
from n4net.utils.gpu_mem import print_gpu_stats,print_peak_gpu_stats

# -- package imports [to test] --
import lidia
from lidia.model_io import get_lidia_model as get_lidia_model_ntire
from lidia.nl_model_io import get_lidia_model as get_lidia_model_nl
from lidia.data import save_burst

# -- check if reordered --
from scipy import optimize
MAX_NFRAMES = 85
DATA_DIR = Path("./data/")
SAVE_DIR = Path("./output/tests/test_same_lidia/")
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir(parents=True)

#
#
# -- Test original LIDIA v.s. modular (n4net) LIDIA --
#
#

@pytest.mark.skip()
def test_same_lidia():

    # -- params --
    sigma = 50.
    device = "cuda:0"
    ps = 5
    vid_set = "toy"
    vid_name = "text_tourbus"

    # -- video --
    vid_cfg = data_hub.get_video_cfg(vid_set,vid_name)
    clean = data_hub.load_video(vid_cfg)[:3,:,:96,:128]
    clean = th.from_numpy(clean).contiguous().to(device)

    # -- set seed --
    seed = 123
    th.manual_seed(seed)
    np.random.seed(seed)

    # -- over training --
    for train in [True,False]:

        # -- get data --
        noisy = clean + sigma * th.randn_like(clean)
        im_shape = noisy.shape

        # -- lidia exec --
        lidia_model = get_lidia_model_nl(device,im_shape,sigma)
        deno_steps = lidia_model.run_parts(noisy,sigma,train=train)
        deno_steps = deno_steps.detach()

        # -- n4net exec --
        n4_model = n4net.lidia.load_model(sigma)
        deno_n4 = n4_model(noisy,sigma,train=train)
        deno_n4 = deno_n4.detach()

        # -- test --
        error = th.sum((deno_n4 - deno_steps)**2).item()
        assert error < 1e-10
#
#
# -- Test modular (n4net) LIDIA [same as og] v.s. diffy (n4net) LIDIA --
#
#

@pytest.mark.skip()
def test_batched():

    # -- params --
    sigma = 50.
    device = "cuda:0"
    ps = 5
    vid_set = "toy"
    vid_name = "text_tourbus"
    gpu_stats = False

    # -- set seed --
    seed = 123
    th.manual_seed(seed)
    np.random.seed(seed)
    th.cuda.set_device(0)

    # -- video --
    vid_cfg = data_hub.get_video_cfg(vid_set,vid_name)
    clean = data_hub.load_video(vid_cfg)[:3,:,:128,:128]
    # clean = data_hub.load_video(vid_cfg)[:3,:,:96,:96]
    clean = th.from_numpy(clean).contiguous().to(device)

    # -- gpu info --
    print_peak_gpu_stats(gpu_stats,"Init.")

    # -- over training --
    for train in [False]:#,False]:

        # -- get data --
        noisy = clean + sigma * th.randn_like(clean)
        im_shape = noisy.shape
        noisy = noisy.contiguous()

        # -- n4net exec --
        n4_model = n4net.lidia.load_model(sigma).to(device)
        deno_steps = n4_model(noisy,sigma,train=train)
        print_gpu_stats(gpu_stats,"post-Step")
        # with th.no_grad():
        #     deno_steps = n4_model(noisy,sigma,train=train)
        deno_steps = deno_steps.detach()/255.

        # -- gpu info --
        print_gpu_stats(gpu_stats,"post-Step.")
        print_peak_gpu_stats(gpu_stats,"post-Step.")

        # -- n4net exec --
        n4b_model = n4net.batched_lidia.load_model(sigma).to(device)
        deno_n4 = n4b_model(noisy,sigma,train=train)
        print_gpu_stats(gpu_stats,"post-Batched.")
        print_peak_gpu_stats(gpu_stats,"post-Batched.")
        # with th.no_grad():
        #     deno_n4 = n4b_model(noisy,sigma,train=train)
        deno_n4 = deno_n4.detach()/255.

        # -- save --
        dnls.testing.data.save_burst(deno_n4,SAVE_DIR,"batched")
        dnls.testing.data.save_burst(deno_steps,SAVE_DIR,"ref")
        diff = th.abs(deno_steps - deno_n4)
        diff /= diff.max()
        dnls.testing.data.save_burst(diff,SAVE_DIR,"diff")

        # -- test --
        error = th.sum((deno_n4 - deno_steps)**2).item()
        assert error < 20. # allow for batch-norm artifacts

        # -- gpu info --
        print_gpu_stats(gpu_stats,"final.")
        print_peak_gpu_stats(gpu_stats,"final.")

#
#
# -- Test internal adaptations for LIDIA and BatchedLIDIA --
#
#

def test_internal_adapt():

    # -- params --
    sigma = 50.
    device = "cuda:0"
    ps = 5
    vid_set = "toy"
    vid_name = "text_tourbus"
    gpu_stats = False

    # -- set seed --
    seed = 123
    th.manual_seed(seed)
    np.random.seed(seed)
    th.cuda.set_device(0)

    # -- video --
    vid_cfg = data_hub.get_video_cfg(vid_set,vid_name)
    clean = data_hub.load_video(vid_cfg)[:3,:,:128,:128]
    # clean = data_hub.load_video(vid_cfg)[:3,:,:96,:96]
    clean = th.from_numpy(clean).contiguous().to(device)

    # -- over training --
    for train in [False]:#,False]:

        # -- get data --
        noisy = clean + sigma * th.randn_like(clean)
        im_shape = noisy.shape
        noisy = noisy.contiguous()

        # -- n4net exec --
        n4_model = n4net.lidia.load_model(sigma).to(device)
        deno_steps = n4_model(noisy,sigma,train=train)
        # with th.no_grad():
        #     deno_steps = n4_model(noisy,sigma,train=train)
        deno_steps = deno_steps.detach()/255.

        # -- n4net exec --
        n4b_model = n4net.batched_lidia.load_model(sigma).to(device)
        deno_n4 = n4b_model(noisy,sigma,train=train)
        print_gpu_stats(gpu_stats,"post-bL")
        # with th.no_grad():
        #     deno_n4 = n4b_model(noisy,sigma,train=train)
        deno_n4 = deno_n4.detach()/255.

        # -- save --
        dnls.testing.data.save_burst(deno_n4,SAVE_DIR,"batched")
        dnls.testing.data.save_burst(deno_steps,SAVE_DIR,"ref")
        diff = th.abs(deno_steps - deno_n4)
        diff /= diff.max()
        dnls.testing.data.save_burst(diff,SAVE_DIR,"diff")

        # -- test --
        error = th.sum((deno_n4 - deno_steps)**2).item()
        print(error)
        assert error < 20. # allow for batch-norm artifacts
