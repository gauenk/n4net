

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
        self.weights_net0 = FcNet()
        self.alpha0 = nn.Parameter(th.tensor((0.5,), dtype=th.float32))

        # -- sep 1 --
        self.weights_net1 = FcNet()
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
        wpatches = patches_n[None,:] * weights
        # h,w = params0['pixels_h'],params0['pixels_w']
        vid,x_out = self.sep_net.run_batched_sep0_a(wpatches,inds,
                                              fold_nl,wfold_nl)
        # vid,wvid = self.sep_net.run_batched_sep0_a(patches_n[None,:],inds,
        #                                            fold_nl,wfold_nl)

        pairs = {"sep0":x_out}
        for key,val in pairs.items():
            print("Key: %s" % key)
            print("val.shape: ",val.shape)
            if "wp" in key:
                val = rearrange(val,'1 (t m) k d -> t m k d',t=3)
            elif "sep" in key:
                val = rearrange(val,'1 (t m) k d -> t m k d',t=3)
            fn = "gt_%s.pt" % key
            gt_val = th.load(fn)
            print(gt_val.shape)
            print(val.shape)
            error = th.sum((val - gt_val)**2).item()
            assert error < 1e-10

        return vid

    def batched_fwd_a1(self,patches_n,dist,inds,fold_nl,wfold_nl):
        weights = self.weights_net1(th.exp(-self.alpha1.abs() * dist))
        weights = weights.unsqueeze(-1)
        wpatches = patches_n * weights
        weights = weights[:, :, 0:1, :]
        # h,w = params1['pixels_h'],params1['pixels_w']
        vid,x_out = self.sep_net.run_batched_sep1_a(wpatches,weights,inds,
                                                     fold_nl,wfold_nl)
        # vid,wvid = self.sep_net.run_batched_sep1_a(patches_n[None,:],weights,inds,
        #                                            fold_nl,wfold_nl)

        pairs = {"sep1":x_out}
        for key,val in pairs.items():
            print("Key: %s" % key)
            print("val.shape: ",val.shape)
            if "wp" in key:
                val = rearrange(val,'1 (t m) k d -> t m k d',t=3)
            elif "sep" in key:
                val = rearrange(val,'1 (t m) k d -> t m k d',t=3)
            fn = "gt_%s.pt" % key
            gt_val = th.load(fn)
            print(gt_val.shape)
            print(val.shape)
            error = th.sum((val - gt_val)**2).item()
            assert error < 1e-10

        return vid

    def batched_fwd_a(self,patches_n0,dist0,inds0,fold_nl0,wfold_nl0,
                      patches_n1,dist1,inds1,fold_nl1,wfold_nl1):

        #
        # -- Sep @ 0 --
        #

        weights0 = self.weights_net0(th.exp(-self.alpha0.abs() * dist0))
        weights0 = weights0.unsqueeze(-1)
        wpatches0 = patches_n0[None,:] * weights0
        # h,w = params0['pixels_h'],params0['pixels_w']
        vid0,sep0 = self.sep_net.run_batched_sep0_a(wpatches0,inds0,
                                               fold_nl0,wfold_nl0)
        # vid0,wvid0 = self.sep_net.run_batched_sep0_a(patches_n0[None,:],inds0,
        #                                              fold_nl0,wfold_nl0)

        #
        # -- Sep @ 1 --
        #

        weights1 = self.weights_net1(th.exp(-self.alpha1.abs() * dist1))
        weights1 = weights1.unsqueeze(-1)
        wpatches1 = patches_n1 * weights1
        weights1 = weights1[:, :, 0:1, :]
        # h,w = params1['pixels_h'],params1['pixels_w']
        # vid1,wvid1 = self.sep_net.run_batched_sep1_a(wpatches1,weights1,inds1,
        #                                              fold_nl1,wfold_nl1)
        vid1 = self.sep_net.run_batched_sep1_a(patches_n1[None,:],weights1,inds1,
                                               fold_nl1,wfold_nl1)

        return vid0,vid1

    def batched_fwd_b(self,patches_n0,dist0,inds0,vid0,wvid0,scatter_nl0,
                      patches_n1,dist1,inds1,vid1,wvid1,scatter_nl1):

        #
        # -- Sep @ 0 --
        #

        weights0 = self.weights_net0(th.exp(-self.alpha0.abs() * dist0))
        weights0 = weights0.unsqueeze(-1)
        wpatches0 = patches_n0 * weights0
        agg0,sep0 = self.sep_net.run_batched_sep0_b(wpatches0,vid0,wvid0,
                                                    inds0,scatter_nl0)
        #
        # -- Sep @ 1 --
        #

        weights1 = self.weights_net1(th.exp(-self.alpha1.abs() * dist1))
        weights1 = weights1.unsqueeze(-1)
        wpatches1 = patches_n1 * weights1
        weights1 = weights1[:, :, 0:1, :]
        # h,w = params1['pixels_h'],params1['pixels_w']
        agg1,sep1 = self.sep_net.run_batched_sep1_b(wpatches1,weights1,vid1,wvid1,
                                                    inds1,scatter_nl1)

        # -- check --
        assert th.any(th.isnan(agg0)).item() is False
        assert th.any(th.isnan(sep0)).item() is False
        assert th.any(th.isnan(agg1)).item() is False
        assert th.any(th.isnan(sep1)).item() is False

        # -- save --
        # th.save(rearrange(wpatches0,'1 (t m) k d -> t m k d',t=3),"bt_wp0.pt")
        # th.save(rearrange(wpatches1,'1 (t m) k d -> t m k d',t=3),"bt_wp1.pt")
        # th.save(sep0,"bt_sep0.pt")
        # th.save(agg0,"bt_agg0.pt")
        # th.save(sep1,"bt_sep1.pt")
        # th.save(agg1,"bt_agg1.pt")

        sep0 = th.load("gt_sep0.pt")
        agg0 = th.load("gt_agg0.pt")
        sep1 = th.load("gt_sep1.pt")
        agg1 = th.load("gt_agg1.pt")

        # pairs = {"wp0":wpatches0,"wp1":wpatches1,
        #          "sep0":sep0,"agg0":agg0,"agg1":agg1,"sep1":sep1}
        # for key,val in pairs.items():
        #     print("Key: %s" % key)
        #     print("val.shape: ",val.shape)
        #     if "wp" in key:
        #         val = rearrange(val,'1 (t m) k d -> t m k d',t=3)
        #     elif "sep" in key:
        #         val = rearrange(val,'(t m) 1 k d -> t m k d',t=3)
        #     elif "agg" in key:
        #         val = rearrange(val,'(t m) 1 k d -> t m k d',t=3)
        #     fn = "gt_%s.pt" % key
        #     gt_val = th.load(fn)
        #     if "agg" in key:
        #         print("val.shape: ",val.shape)
        #         ps = 5
        #         shape_s = 't (h w) k d -> t h w k d'
        #         gt_val = rearrange(gt_val,shape_s,h=100)
        #         val = rearrange(val,shape_s,h=100)
        #         # shape_s = 't (h w) k (c ph pw) -> t h w k c ph pw'
        #         # gt_val = rearrange(gt_val,shape_s,h=100,ph=ps,pw=ps)
        #         # val = rearrange(val,shape_s,h=100,ph=ps,pw=ps)
        #         print("-"*10)
        #         for i in range(10):
        #             print("-"*10)
        #             print(gt_val[0,i,i,0,:5])
        #             print(val[0,i,i,0,:5])
        #         print("-"*10)
        #     error = th.sum((val - gt_val)**2).item()
        #     print(error)
        #     assert error < 1e-1

        #
        # -- Final Sep --
        #

        # print("agg0.shape: ",agg0.shape)
        # print("sep0.shape: ",sep0.shape)
        # print("agg1.shape: ",agg1.shape)
        # print("sep1.shape: ",sep1.shape)
        inputs = th.cat((sep0, sep1, agg0, agg1), dim=-2)

        print("inputs.shape: ",inputs.shape)
        # inputs = rearrange(inputs,'(t n) 1 f d -> t n f d',t=3)
        # print("inputs.shape: ",inputs.shape)
        noise = self.sep_net.sep_part2(inputs)

        # print("patches_n0.shape: ",patches_n0.shape)
        patches_n0 = rearrange(patches_n0,'(t n) k d -> t n k d',t=3)
        # print("patches_n0.shape: ",patches_n0.shape)
        # print("noise.shape: ",noise.shape)
        deno,patches_w = self.run_batched_pdn_final(patches_n0,noise)
        # print("deno.shape: ",deno.shape)
        # print("patches_w.shape: ",patches_w.shape)

        # -- rearrange --
        # deno = rearrange(deno,'t n 1 d -> (t n) 1 d')
        # patches_w = rearrange(patches_w,'t n 1 d -> (t n) 1 d')
        # print("deno.shape: ",deno.shape)
        # print("patches_w.shape: ",patches_w.shape)

        return deno,patches_w

    def run_batched_pdn_final(self,patches_n0,noise):
        patches_dn = patches_n0[:, :, 0, :] - noise.squeeze(-2)
        patches_no_mean = patches_dn - patches_dn.mean(dim=-1, keepdim=True)
        patch_exp_weights = (patches_no_mean ** 2).mean(dim=-1, keepdim=True)
        patch_weights = th.exp(-self.beta.abs() * patch_exp_weights)
        return patches_dn,patch_weights

        # patches_dn = patches_n0[:, :, [0], :] - noise[:,:]
        # patches_no_mean = patches_dn - patches_dn.mean(dim=-1, keepdim=True)
        # patch_exp_weights = (patches_no_mean ** 2).mean(dim=-1, keepdim=True)
        # patch_weights = th.exp(-self.beta.abs() * patch_exp_weights)
        # return patches_dn,patch_weights

