
# -- linalg --
import torch as th
from einops import rearrange,repeat

# -- management --
from easydict import EasyDict as edict

# -- submodules --
from .pdn_structs import PatchDenoiseNet

# -- neural network --
import torch.nn as nn
import torch.nn.functional as nn_func
from torch.nn.functional import fold
from torch.nn.functional import pad as nn_pad
from torchvision.transforms.functional import center_crop

# -- diff. non-local search --
import dnls

# -- separate logic --
from . import nn_impl
from . import im_shapes

# -- utils --
from n4net.utils import clean_code

# -- misc imports --
from .misc import crop_offset,get_npatches,assert_nonan,get_step_fxns

@clean_code.add_methods_from(im_shapes)
@clean_code.add_methods_from(nn_impl)
class BatchedLIDIA(nn.Module):

    def __init__(self, pad_offs, arch_opt, lidia_pad=True):
        super(BatchedLIDIA, self).__init__()
        self.arch_opt = arch_opt
        self.pad_offs = pad_offs
        self.lidia_pad = lidia_pad

        self.patch_w = 5 if arch_opt.rgb else 7
        self.ps = self.patch_w
        self.neigh_pad = 14
        self.ver_size = 80 if arch_opt.rgb else 64

        self.rgb2gray = nn.Conv2d(in_channels=3, out_channels=1,
                                  kernel_size=(1, 1), bias=False)
        self.rgb2gray.weight.data = th.tensor([0.2989, 0.5870, 0.1140],
                                                 dtype=th.float32).view(1, 3, 1, 1)
        self.rgb2gray.weight.requires_grad = False

        kernel_1d = th.tensor((1 / 4, 1 / 2, 1 / 4), dtype=th.float32)
        kernel_2d = (kernel_1d.view(-1, 1) * kernel_1d).view(1, 1, 3, 3)
        self.bilinear_conv = nn.Conv2d(in_channels=1, out_channels=1,
                                       kernel_size=(3, 3), bias=False)
        self.bilinear_conv.weight.data = kernel_2d
        self.bilinear_conv.weight.requires_grad = False

        self.pdn = PatchDenoiseNet(arch_opt=arch_opt,patch_w=self.patch_w,
                                   ver_size=self.ver_size)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #          Forward Pass
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def forward(self, noisy, sigma, train=False, srch_img=None, flows=None,
                rescale=True, ws=29, wt=0, stride=1, batch_size = -1):
        """

        Primary Network Backbone

        """

        #
        # -- Prepare --
        #

        # -- normalize for input ---
        if rescale: noisy = (noisy/255. - 0.5)/0.5
        means = noisy.mean((-2,-1),True)
        noisy -= means
        if srch_img is None:
            srch_img = noisy
        noisy = noisy.contiguous()

        # -- unpack --
        device = noisy.device
        t,c,h,w = noisy.shape
        ps,pt = self.patch_w,1
        hp,wp = h+2*(ps//2),w+2*(ps//2)

        # -- patch-based functions --
        levels = self.get_levels()#{"l0":{"dil":1},"l1":{"dil":2}}
        pfxns = edict()
        for lname,params in levels.items():
            dil = params['dil']
            h_l,w_l,pad_l = self.image_shape((hp,wp),ps,dilation=dil)
            coords_l = [pad_l,pad_l,hp+pad_l,hp+pad_l]
            vshape_l = (t,c,h_l,w_l)
            pfxns[lname] = get_step_fxns(vshape_l,coords_l,ps,stride,dil,device)

        #
        # -- First Processing --
        #

        # -- Loop Info --
        t,c,h,w = noisy.shape
        nqueries = t * ((hp-1)//stride+1) * ((wp-1)//stride+1)
        if batch_size <= 0: batch_size = nqueries
        # batch_size = 128
        # batch_size = nqueries//4
        # batch_size = nqueries//2
        nbatches = (nqueries - 1)//batch_size+1

        for batch in range(nbatches):

            # -- Batching Inds --
            qindex = min(batch * batch_size,nqueries)
            batch_size = min(batch_size,nqueries - qindex)
            queries = dnls.utils.inds.get_query_batch(qindex,batch_size,
                                                      stride,t,hp,wp,device)
            # -- Process Each Level --
            for level in levels:
                nn_fxn = levels[level]['nn_fxn']
                pdn_fxn = levels[level]['pdn_fxn']
                self.first_step(noisy,srch_img,pfxns[level],flows,sigma,
                                ws,wt,qindex,queries,nn_fxn,pdn_fxn,train)


        #
        # -- Normalize Videos --
        #

        for level in levels:
            vid = pfxns[level].fold_nl.vid
            wvid = pfxns[level].wfold_nl.vid
            vid_z = vid / wvid
            assert_nonan(vid)
            assert_nonan(wvid)
            assert_nonan(vid_z)
            levels[level]['vid'] = vid_z

        # -- decl fxns --
        pad = self.ps//2
        coordsF = [0,0,hp,wp]
        fold_nl = dnls.ifold.iFold((t,c,hp,wp),coordsF,stride=1,dilation=1)
        wfold_nl = dnls.ifold.iFold((t,c,hp,wp),coordsF,stride=1,dilation=1)
        unfold0 = dnls.iunfold.iUnfold(ps,pfxns.l0.fold_nl.coords,stride=1,dilation=1)
        unfold1 = dnls.iunfold.iUnfold(ps,pfxns.l1.fold_nl.coords,stride=1,dilation=2)
        scatter0 = pfxns.l0.scatter
        scatter1 = pfxns.l1.scatter
        vid0 = levels["l0"]['vid']
        vid1 = levels["l1"]['vid']

        # -- second step --
        for batch in range(nbatches):

            #
            # -- Batching Inds --
            #

            qindex = min(batch * batch_size,nqueries)
            batch_size = min(batch_size,nqueries - qindex)
            queries = dnls.utils.inds.get_query_batch(qindex,batch_size,
                                                      stride,t,hp,wp,device)
            unfold0.qnum = batch_size
            unfold1.qnum = batch_size

            #
            # -- Non-Local Search --
            #

            # -- [nn0 search]  --
            output0 = self.run_nn0(noisy,queries,scatter0,srch_img,
                                   flows,train,ws=ws,wt=wt)
            patches0 = output0[0]
            dists0 = output0[1]
            inds0 = output0[2]
            params0 = output0[3]

            # -- [nn1 search]  --
            output1 = self.run_nn1(noisy,queries,scatter1,srch_img,
                                   flows,train,ws=ws,wt=wt)
            patches1 = output1[0]
            dists1 = output1[1]
            inds1 = output1[2]
            params1 = output1[3]

            #
            # -- Patch-based Denoising --
            #

            # -- reshape --
            dists0 = rearrange(dists0,'n k -> 1 n k')
            dists1 = rearrange(dists1,'n k -> 1 n k')
            inds0 = inds0[:,[0]]
            inds1 = inds1[:,[0]]

            # -- checks --
            assert th.any(th.isnan(patches0)).item() is False
            assert th.any(th.isnan(patches1)).item() is False
            assert th.any(th.isnan(inds0)).item() is False
            assert th.any(th.isnan(inds1)).item() is False

            # -- for now --
            inds0 = qindex
            inds1 = qindex

            # -- exec --
            outs = self.pdn.batched_fwd_b(patches0,dists0,inds0,vid0,unfold0,
                                          patches1,dists1,inds1,vid1,unfold1)
            pdeno,patches_w = outs

            #
            # -- Final Weight Aggregation --
            #
            assert_nonan(pdeno)
            assert_nonan(patches_w)

            self.run_parts_final(pdeno,patches_w,inds0,params0,
                                 qindex,fold_nl,wfold_nl)

        #
        # -- Final Format --
        #

        # -- unpack --
        deno = self.final_format(fold_nl,wfold_nl)
        assert_nonan(deno)

        # -- normalize for output ---
        deno += means # normalize
        noisy += means # restore
        if rescale:
            deno[...]  = 255.*(deno  * 0.5 + 0.5) # normalize
            noisy[...] = 255.*(noisy * 0.5 + 0.5) # restore
        return deno

    def run_parts_final(self,image_dn,patch_weights,inds,params,
                        qindex,fold_nl,wfold_nl):

        # -- expands wpatches --
        pdim = image_dn.shape[-1]
        image_dn = image_dn * patch_weights
        ones_tmp = th.ones(1, 1, pdim, device=image_dn.device)
        wpatches = (patch_weights * ones_tmp)

        # -- format to fold --
        ps = self.patch_w
        shape_str = 't n (c h w) -> (t n) 1 1 c h w'
        image_dn = rearrange(image_dn,shape_str,h=ps,w=ps)
        wpatches = rearrange(wpatches,shape_str,h=ps,w=ps)

        # -- contiguous --
        image_dn = image_dn.contiguous()
        wpatches = wpatches.contiguous()

        # -- dnls fold --
        image_dn = fold_nl(image_dn,qindex)
        patch_cnt = wfold_nl(wpatches,qindex)

    def final_format(self,fold_nl,wfold_nl):
        # -- crop --
        pad = self.ps//2
        image_dn = fold_nl.vid
        patch_cnt = wfold_nl.vid
        image_dn = image_dn[:,:,pad:-pad,pad:-pad]
        image_dn /= patch_cnt[:,:,pad:-pad,pad:-pad]
        return image_dn

    def get_levels(self):
        levels = {"l0":{"dil":1,
                        "nn_fxn":self.run_nn0,
                        "pdn_fxn":self.pdn.batched_fwd_a0},
                  "l1":{"dil":2,
                        "nn_fxn":self.run_nn1,
                        "pdn_fxn":self.pdn.batched_fwd_a1},
        }
        return levels

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #       One Level of First Step
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def first_step(self,noisy,srch_img,pfxns,flows,sigma,
                   ws,wt,qindex,queries,nn_fxn,pdn_fxn,train):

        # -- Non-Local Search --
        output = nn_fxn(noisy,queries,pfxns.scatter,
                        srch_img,flows,train,
                        ws=ws,wt=wt)
        patches = output[0]
        dists = output[1]
        inds = output[2]
        params = output[3]

        # -- Patch-based Denoising --
        dists = rearrange(dists,'n k -> 1 n k')
        inds = qindex
        outs = pdn_fxn(patches,dists,inds,pfxns.fold_nl,pfxns.wfold_nl)

        return outs

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #         Padding & Cropping
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- pad/crops --
    def pad_crop0(self):
        raise NotImplemented("")

    def _pad_crop0_eff():
        raise NotImplemented("")

    def _pad_crop0_og():
        raise NotImplemented("")

    def pad_crop1(self):
        raise NotImplemented("")

    def _pad_crop1(self):
        raise NotImplemented("")

    # -- shapes --
    def image_n0_shape(self):
        raise NotImplemented("")

    def image_n0_shape_og(self):
        raise NotImplemented("")

    def image_n1_shape(self):
        raise NotImplemented("")

    def image_n1_shape_og(self):
        raise NotImplemented("")

    # -- prepare --
    def prepare_image_n1(self):
        raise NotImplemented("")

    def prepare_image_n1_eff(self):
        raise NotImplemented("")

    def prepare_image_n1_og(self):
        raise NotImplemented("")

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #      Nearest Neighbor Searches
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def run_nn0(self):
        raise NotImplemented("")

    def run_nn1(self):
        raise NotImplemented("")

class ArchitectureOptions:
    def __init__(self, rgb):
        self.rgb = rgb

