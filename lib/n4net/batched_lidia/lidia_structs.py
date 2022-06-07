
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
from torchvision.transforms.functional import center_crop

# -- diff. non-local search --
import dnls

# -- separate logic --
from . import nn_impl

# -- utils --
from n4net.utils import clean_code

# -- misc imports --
from .misc import crop_offset,get_npatches,assert_nonan

@clean_code.add_methods_from(nn_impl)
class BatchedLIDIA(nn.Module):

    def __init__(self, pad_offs, arch_opt):
        super(BatchedLIDIA, self).__init__()
        self.arch_opt = arch_opt
        self.pad_offs = pad_offs

        self.patch_w = 5 if arch_opt.rgb else 7
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

        # -- iparams --
        i0_shape = self.image_n0_shape(noisy,train)
        i1_shape = self.image_n1_shape(noisy,train)
        _,_,h0,w0 = i0_shape
        _,_,h1,w1 = i1_shape
        _,_,h,w = noisy.shape
        dil = 1
        pad = ps//2 + dil*(ps//2)
        h0,w0 = h+2*pad,w+2*pad
        dil = 2
        pad = ps//2 + dil*(ps//2)
        h1,w1 = h+2*pad,w+2*pad

        # -- get coords for fill image: (top,left,btm,right) --
        hp,wp = h+2*(ps//2),w+2*(ps//2)
        hpad,wpad = (h0 - hp)//2,(w0 - wp)//2
        coords0 = (hpad,wpad,hp+hpad,wp+wpad)
        hpad,wpad = (h1 - hp)//2,(w1 - wp)//2
        coords1 = (hpad,wpad,hp+hpad,wp+wpad)
        vshape0 = (t,c,h0,w0)
        vshape1 = (t,c,h1,w1)

        #
        # -- First Step --
        #

        # -- Loop Args --
        t,c,h,w = noisy.shape
        nqueries = t * ((hp-1)//stride+1) * ((wp-1)//stride+1)
        if batch_size <= 0: batch_size = nqueries
        # batch_size = 128
        # batch_size = nqueries//4
        # batch_size = nqueries//2
        nbatches = (nqueries - 1)//batch_size+1

        # -- Scatter/Fold Fxns --
        dil0,dil1 = 1,2
        p0_fxns = self.get_patch_fxns(vshape0,coords0,stride,dil0,device)
        p1_fxns = self.get_patch_fxns(vshape1,coords1,stride,dil1,device)

        for batch in range(nbatches):

            # -- Batching Inds --
            qindex = min(batch * batch_size,nqueries)
            batch_size = min(batch_size,nqueries - qindex)
            queries = dnls.utils.inds.get_query_batch(qindex,batch_size,
                                                      stride,t,hp,wp,device)
            # -- Level 0 --
            vid0 = self.first_step(noisy,srch_img,p0_fxns,flows,sigma,
                                   vshape0,ws,wt,qindex,queries,
                                   self.run_nn0,self.pdn.batched_fwd_a0,train)
            # -- Level 1 --
            vid1 = self.first_step(noisy,srch_img,p1_fxns,flows,sigma,
                                   vshape1,ws,wt,qindex,queries,
                                   self.run_nn1,self.pdn.batched_fwd_a1,train)

        # -- unpack --
        vid0,wvid0 = p0_fxns.fold_nl.vid,p0_fxns.wfold_nl.vid
        vid1,wvid1 = p1_fxns.fold_nl.vid,p1_fxns.wfold_nl.vid
        vid0 = vid0 / wvid0
        vid1 = vid1 / wvid1

        # -- normalize --
        # dnls.testing.data.save_burst(vid0,"./output/tests/","vid0")
        # dnls.testing.data.save_burst(vid1,"./output/tests/","vid1")

        # -- checks --
        assert th.any(th.isnan(vid0)).item() is False
        assert th.any(th.isnan(wvid0)).item() is False
        assert th.any(th.isnan(vid1)).item() is False
        assert th.any(th.isnan(wvid1)).item() is False

        # -- decl fxns --
        coords0 = [0,0,hp,wp]
        coords0 = [2,2,hp+2,wp+2]
        _hp,_wp = h+2*(ps-1),w+2*(ps-1)
        fold_nl = dnls.ifold.iFold((t,c,_hp,_wp),coords0,stride=1,dilation=1)
        wfold_nl = dnls.ifold.iFold((t,c,_hp,_wp),coords0,stride=1,dilation=1)
        unfold0 = dnls.iunfold.iUnfold(ps,coords0,stride=1,dilation=1)
        unfold1 = dnls.iunfold.iUnfold(ps,coords1,stride=1,dilation=2)
        scatter0 = p0_fxns.scatter
        scatter1 = p1_fxns.scatter

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
            output0 = self.run_nn0(noisy,queries,scatter0,
                                   srch_img,flows,train,
                                   ws=ws,wt=wt)
            patches0 = output0[0]
            dists0 = output0[1]
            inds0 = output0[2]
            params0 = output0[3]

            # -- [nn1 search]  --
            output1 = self.run_nn1(noisy,queries,scatter1,
                                   srch_img,flows,train,
                                   ws=ws,wt=wt)
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
        # -- Format --
        #

        # -- unpack --
        deno = self.final_format(fold_nl,wfold_nl,_hp,_wp)
        assert th.any(th.isnan(deno)).item() is False

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

        # -- fold --
        # h,w = params['pixels_h'],params['pixels_w']
        # shape = (h,w)
        # image_dn = fold(image_dn,shape,(ps,ps))
        # patch_cnt = fold(wpatches,shape,(ps,ps))

        # -- dnls fold --
        image_dn = fold_nl(image_dn,qindex)
        patch_cnt = wfold_nl(wpatches,qindex)

    def final_format(self,fold_nl,wfold_nl,hp,wp):
        # -- crop --
        ps = self.patch_w
        image_dn = fold_nl.vid
        patch_cnt = wfold_nl.vid
        row_offs = min(ps - 1, hp - 1)
        col_offs = min(ps - 1, wp - 1)
        image_dn = crop_offset(image_dn, (row_offs,), (col_offs,))
        image_dn /= crop_offset(patch_cnt, (row_offs,), (col_offs,))
        return image_dn

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #       One Level of First Step
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def get_patch_fxns(self,vshape,coords,stride,dilation,device):
        ps,pt,dil = self.patch_w,1,dilation
        scatter = dnls.scatter.ScatterNl(ps,pt,dilation=dil,exact=True)
        fold_nl = dnls.ifold.iFold(vshape,coords,stride=stride,dilation=dil)
        wfold_nl = dnls.ifold.iFold(vshape,coords,stride=stride,dilation=dil)
        pfxns = edict()
        pfxns.scatter = scatter
        pfxns.fold_nl = fold_nl
        pfxns.wfold_nl = wfold_nl
        return pfxns

    def first_step(self,noisy,srch_img,pfxns,flows,sigma,vshape,
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
    #           Padding & Cropping
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def pad_crop0(self, image, pad_offs, train):
        return self._pad_crop0(image, pad_offs, train, self.patch_w)

    @staticmethod
    def _pad_crop0(image,pad_offs,train,patch_w):
        if not train:
            reflect_pad = [patch_w - 1] * 4
            constant_pad = [14] * 4
            image = nn_func.pad(nn_func.pad(image, reflect_pad, 'reflect'),
                                constant_pad, 'constant', -1)
        else:
            image = crop_offset(image, (pad_offs,), (pad_offs,))
        return image

    def pad_crop1(self, image, train, mode):
        return self._pad_crop1(image, train, mode, self.patch_w)

    @staticmethod
    def _pad_crop1(image, train, mode, patch_w):
        if not train:
            if mode == 'reflect':
                bilinear_pad = 1
                averaging_pad = (patch_w - 1) // 2
                patch_w_scale_1 = 2 * patch_w - 1
                find_nn_pad = (patch_w_scale_1 - 1) // 2
                reflect_pad = [averaging_pad + bilinear_pad + find_nn_pad] * 4
                image = nn_func.pad(image, reflect_pad, 'reflect')
            elif mode == 'constant':
                constant_pad = [28] * 4
                image = nn_func.pad(image, constant_pad, 'constant', -1)
            else:
                assert False
        return image

    def image_n0_shape(self,image_n,train,k=14):
        pad_offs = self.pad_offs
        ps = self.patch_w

        if not train:
            reflect_pad = ps - 1
            constant_pad = k
            pad = reflect_pad + constant_pad
            t,c,h,w = image_n.shape
            hp = h + 2*pad
            wp = w + 2*pad
            return t,c,hp,wp
        else:
            t,c,h,w = image_n.shape
            hp = h - pad_offs
            wp = w - pad_offs
            return t,c,hp,wp


    def image_n1_shape(self,image_n,train,k=14):
        patch_w = self.patch_w
        if train:
            t,c,h,w = image_n.shape
            hp,wp = h-2,w-2
            return t,c,h,w
        else:
            bilinear_pad = 1
            averaging_pad = (patch_w - 1) // 2
            patch_w_scale_1 = 2 * self.patch_w - 1
            find_nn_pad = (patch_w_scale_1 - 1) // 2
            constant_pad = 2*k
            pad = averaging_pad + find_nn_pad + constant_pad
            t,c,h,w = image_n.shape
            hp,wp = h+2*pad,w+2*pad
            return t,c,hp,wp

    def prepare_image_n1(self,image_n,train):

        # -- pad & unpack --
        patch_numel = (self.patch_w ** 2) * image_n.shape[1]
        device = image_n.device
        image_n1 = self.pad_crop1(image_n, train, 'reflect')
        im_n1_b, im_n1_c, im_n1_h, im_n1_w = image_n1.shape

        # -- bilinear conv & crop --
        # image_n1 = image_n1[:,:,1:-1,1:-1]
        image_n1 = image_n1.view(im_n1_b * im_n1_c, 1,im_n1_h, im_n1_w)
        image_n1 = self.bilinear_conv(image_n1)
        image_n1 = image_n1.view(im_n1_b, im_n1_c, im_n1_h - 2, im_n1_w - 2)
        image_n1 = self.pad_crop1(image_n1, train, 'constant')
        return image_n1

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

