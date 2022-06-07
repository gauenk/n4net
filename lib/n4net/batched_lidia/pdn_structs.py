

# -- linalg --
import torch as th
from einops import rearrange,repeat

# -- neural network --
import torch.nn as nn

# -- submodules --
from .sep_structs import SeparableFcNet,FcNet

class PatchDenoiseNet(nn.Module):
    def __init__(self, arch_opt, patch_w, ver_size):
        super(PatchDenoiseNet, self).__init__()

        # -- options --
        self.arch_opt = arch_opt

        # -- sep filters --
        self.sep_net = SeparableFcNet(arch_opt=arch_opt,
                                      patch_w=patch_w,
                                      ver_size=ver_size)
        # -- sep 0 --
        self.weights_net0 = FcNet("fc0")
        self.alpha0 = nn.Parameter(th.tensor((0.5,), dtype=th.float32))

        # -- sep 1 --
        self.weights_net1 = FcNet("fc1")
        self.alpha1 = nn.Parameter(th.tensor((0.5,), dtype=th.float32))

        # -- combo --
        self.beta = nn.Parameter(th.tensor((0.5,), dtype=th.float32))

    def forward(self,patches_n0,dist0,inds0,params0,patches_n1,dist1,inds1,params1):
        """
        Run patch denoiser network
        """

        #
        # -- Sep @ 0 --
        #

        weights0 = self.weights_net0(th.exp(-self.alpha0.abs() * dist0))
        weights0 = weights0.unsqueeze(-1)
        wpatches0 = patches_n0 * weights0
        h,w = params0['pixels_h'],params0['pixels_w']
        agg0,sep0,_ = self.sep_net.run_sep0(wpatches0,dist0,inds0,h,w)

        #
        # -- Sep @ 1 --
        #

        weights1 = self.weights_net1(th.exp(-self.alpha1.abs() * dist1))
        weights1 = weights1.unsqueeze(-1)
        wpatches1 = patches_n1 * weights1
        weights1 = weights1[:, :, 0:1, :]
        h,w = params1['pixels_h'],params1['pixels_w']
        agg1,sep1,_ = self.sep_net.run_sep1(wpatches1,weights1,dist1,inds1,h,w)
        assert th.any(th.isnan(agg1)).item() is False

        #
        # -- Final Sep --
        #

        inputs = th.cat((sep0, sep1, agg0, agg1), dim=-2)
        noise = self.sep_net.sep_part2(inputs)
        deno,patches_w = self.run_pdn_final(patches_n0,noise)
        return deno,patches_w

    def run_pdn_final(self,patches_n0,noise):
        patches_dn = patches_n0[:, :, 0, :] - noise.squeeze(-2)
        patches_no_mean = patches_dn - patches_dn.mean(dim=-1, keepdim=True)
        patch_exp_weights = (patches_no_mean ** 2).mean(dim=-1, keepdim=True)
        patch_weights = th.exp(-self.beta.abs() * patch_exp_weights)
        return patches_dn,patch_weights

    def batched_fwd_a0(self,patches_n,dist,inds,fold_nl,wfold_nl):
        weights = self.weights_net0(th.exp(-self.alpha0.abs() * dist))
        weights = weights.unsqueeze(-1)
        wpatches = patches_n * weights
        vid,x_out = self.sep_net.run_batched_sep0_a(wpatches,inds,
                                              fold_nl,wfold_nl)
        return vid

    def batched_fwd_a1(self,patches_n,dist,inds,fold_nl,wfold_nl):
        weights = self.weights_net1(th.exp(-self.alpha1.abs() * dist))
        weights = weights.unsqueeze(-1)
        wpatches = patches_n * weights
        weights = weights[:, :, 0:1, :]
        vid,x_out = self.sep_net.run_batched_sep1_a(wpatches,weights,inds,
                                                     fold_nl,wfold_nl)
        return vid

    def batched_fwd_a(self,patches_n0,dist0,inds0,fold_nl0,wfold_nl0,
                      patches_n1,dist1,inds1,fold_nl1,wfold_nl1):

        #
        # -- Sep @ 0 --
        #

        weights0 = self.weights_net0(th.exp(-self.alpha0.abs() * dist0))
        weights0 = weights0.unsqueeze(-1)
        wpatches0 = patches_n0 * weights0
        vid0,sep0 = self.sep_net.run_batched_sep0_a(wpatches0,inds0,
                                               fold_nl0,wfold_nl0)

        #
        # -- Sep @ 1 --
        #

        weights1 = self.weights_net1(th.exp(-self.alpha1.abs() * dist1))
        weights1 = weights1.unsqueeze(-1)
        wpatches1 = patches_n1 * weights1
        weights1 = weights1[:, :, 0:1, :]
        vid1 = self.sep_net.run_batched_sep1_a(patches_n1,weights1,inds1,
                                               fold_nl1,wfold_nl1)

        return vid0,vid1

    def batched_fwd_b(self,patches_n0,dist0,inds0,vid0,scatter_nl0,
                      patches_n1,dist1,inds1,vid1,scatter_nl1):

        #
        # -- Sep @ 0 --
        #

        weights0 = self.weights_net0(th.exp(-self.alpha0.abs() * dist0))
        weights0 = weights0.unsqueeze(-1)
        wpatches0 = patches_n0 * weights0
        agg0,sep0 = self.sep_net.run_batched_sep0_b(wpatches0,vid0,
                                                    inds0,scatter_nl0)
        #
        # -- Sep @ 1 --
        #

        weights1 = self.weights_net1(th.exp(-self.alpha1.abs() * dist1))
        weights1 = weights1.unsqueeze(-1)
        wpatches1 = patches_n1 * weights1
        weights1 = weights1[:, :, 0:1, :]
        agg1,sep1 = self.sep_net.run_batched_sep1_b(wpatches1,weights1,vid1,
                                                    inds1,scatter_nl1)

        # -- check --
        assert th.any(th.isnan(agg0)).item() is False
        assert th.any(th.isnan(sep0)).item() is False
        assert th.any(th.isnan(agg1)).item() is False
        assert th.any(th.isnan(sep1)).item() is False

        #
        # -- Final Sep --
        #

        inputs = th.cat((sep0, sep1, agg0, agg1), dim=-2)
        noise = self.sep_net.sep_part2(inputs)
        deno,patches_w = self.run_batched_pdn_final(patches_n0,noise)

        return deno,patches_w

    def run_batched_pdn_final(self,patches_n0,noise):
        patches_dn = patches_n0[:, :, 0, :] - noise.squeeze(-2)
        patches_no_mean = patches_dn - patches_dn.mean(dim=-1, keepdim=True)
        patch_exp_weights = (patches_no_mean ** 2).mean(dim=-1, keepdim=True)
        patch_weights = th.exp(-self.beta.abs() * patch_exp_weights)
        return patches_dn,patch_weights

