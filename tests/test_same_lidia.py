"""

When first created our model is identical to lidia

"""

# -- misc --
import sys,tqdm
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
import n4net
import dnls # supporting

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
# -- Primary Testing Class --
#
#

class TestSameLidia(unittest.TestCase):

    #
    # -- Load Data --
    #

    def load_burst(self,name,ext="jpg"):
        path = DATA_DIR / name
        assert path.exists()
        burst = []
        for t in range(MAX_NFRAMES):
            fn = path / ("%05d.%s" % (t,ext))
            if not fn.exists(): break
            img_t = Image.open(str(fn)).convert("RGB")
            img_t = np.array(img_t)
            img_t = rearrange(img_t,'h w c -> c h w')
            burst.append(img_t)
        if len(burst) == 0:
            print(f"WARNING: no images loaded. Check ext [{ext}]")
        burst = 1.*np.stack(burst)
        burst = th.from_numpy(burst).type(th.float32)
        return burst

    def skip_test_same_lidia(self):

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
            error < 1e-10

    def test_batched(self):

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
        for train in [True]:#,False]:

            # -- get data --
            noisy = clean + sigma * th.randn_like(clean)
            im_shape = noisy.shape
            noisy = noisy.contiguous()

            # -- lidia exec --
            # lidia_model = get_lidia_model_nl(device,im_shape,sigma)
            # deno_steps = lidia_model.run_parts(noisy,sigma,train=train)
            # deno_steps = deno_steps.detach()

            # -- n4net exec --
            n4_model = n4net.lidia.load_model(sigma)
            deno_steps = n4_model(noisy,sigma,train=train)
            deno_steps = deno_steps.detach()

            # -- n4net exec --
            n4b_model = n4net.batched_lidia.load_model(sigma)
            deno_n4 = n4b_model(noisy,sigma,train=train)
            deno_n4 = deno_n4.detach()

            # -- save --
            deno_n4 /= deno_n4.max()
            deno_steps /= deno_steps.max()
            dnls.testing.data.save_burst(deno_n4,SAVE_DIR,"batched")
            dnls.testing.data.save_burst(deno_steps,SAVE_DIR,"ref")

            # -- test --
            error = th.sum((deno_n4 - deno_steps)**2).item()
            error < 1e-10


