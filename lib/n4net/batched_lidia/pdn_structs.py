

# -- linalg --
import torch as th
from einops import rearrange,repeat

# -- neural network --
import torch.nn as nn

# -- submodules --
from .sep_structs import SeparableFcNet,FcNet

# -- checking --
from .misc import assert_nonan

class PatchDenoiseNet(nn.Module):
    def __init__(self, arch_opt, patch_w, ver_size):
        super(PatchDenoiseNet, self).__init__()

        # -- options --
        self.arch_opt = arch_opt
        self.gpu_stats = True

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
        assert_nonan(agg1)

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

    def batched_fwd_a0(self,patches_n,dist,inds,fold,wfold,wdiv=False):
        weights = self.weights_net0(th.exp(-self.alpha0.abs() * dist))
        weights = weights.unsqueeze(-1)
        wpatches = patches_n * weights
        weights = weights[:, :, 0:1, :]
        vid,x_out = self.sep_net.run_batched_sep0_a(wpatches,weights,inds,
                                                    fold,wfold,wdiv)
        return vid

    def batched_fwd_a1(self,patches_n,dist,inds,fold,wfold,wdiv=True):
        weights = self.weights_net1(th.exp(-self.alpha1.abs() * dist))
        weights = weights.unsqueeze(-1)
        wpatches = patches_n * weights
        weights = weights[:, :, 0:1, :]
        vid,x_out = self.sep_net.run_batched_sep1_a(wpatches,weights,inds,
                                                     fold,wfold,wdiv)
        return vid

    def batched_step(self,nn_info,pfxns,params,level,qindex):

        # -- level info --
        alphas = {"l0":self.alpha0,"l1":self.alpha1}
        weight_nets = {"l0":self.weights_net0,"l1":self.weights_net1}
        sep_nets = {"l0":self.sep_net.run_batched_sep0_a,
                    "l1":self.sep_net.run_batched_sep1_a}

        # -- unpack --
        alpha = alphas[level]
        wnet = weight_nets[level]
        sep_net = sep_nets[level]
        patches = nn_info.patches
        dists = nn_info.dists
        wdiv = params.wdiv

        # -- forward --
        weights = wnet(th.exp(-alpha.abs() * dists)).unsqueeze(-1)
        wpatches = patches * weights
        weights = weights[:, :, 0:1, :]
        vid,x_out = sep_net(wpatches,weights,qindex,pfxns,wdiv)
        # vid,x_out = self.sep_net.run_batched_sep1_a(wpatches,weights,inds,
        #                                              fold,wfold,wdiv)
        return vid,x_out

    def batched_fwd_b(self,levels,nn_info,pfxns,qindex,bsize):

        # -- level info --
        alphas = {"l0":self.alpha0,"l1":self.alpha1}
        weight_nets = {"l0":self.weights_net0,"l1":self.weights_net1}
        sep_nets = {"l0":self.sep_net.run_batched_sep0_b,
                    "l1":self.sep_net.run_batched_sep1_b}


        # -- iter each level --
        grouped_sep,grouped_agg = [],[]
        for level in nn_info.keys():

            # -- unpack --
            alpha = alphas[level]
            wnet = weight_nets[level]
            sep_net = sep_nets[level]
            patches = nn_info[level].patches
            dists = nn_info[level].dists
            vid = levels[level].vid
            unfold = pfxns[level].unfold
            wdiv = levels[level].wdiv

            # -- run --
            weights = wnet(th.exp(-alpha.abs() * dists))
            weights = weights.unsqueeze(-1)
            wpatches = patches * weights
            weights = weights[:, :, 0:1, :]
            agg_l,sep_l = sep_net(wpatches,weights,vid,qindex,bsize,unfold,wdiv)
            assert_nonan(agg_l)
            assert_nonan(sep_l)
            grouped_sep.append(sep_l)
            grouped_agg.append(agg_l)

            # -- save mem --
            if level != "l0":
                del patches,nn_info[level].patches
            del nn_info[level].dists,dists
            del weights,wpatches

        # -- gpu info --
        if self.gpu_stats:
            th.cuda.synchronize()
            th.cuda.empty_cache()
            mem_gb = th.cuda.memory_reserved() / 1024**3
            print("[PDN(b)] Reserved GPU Mem (GB): ",mem_gb)
            mem_gb = th.cuda.memory_allocated() / 1024**3
            print("[PDN(b)] Allocated GPU Mem (GB): ",mem_gb)
            # print(th.cuda.memory_summary())

        # -- cat --
        patches_n0 = nn_info['l0'].patches
        grouped = grouped_sep + grouped_agg
        inputs = th.cat(grouped,dim=-2)
        del grouped

        # -- gpu info --
        if self.gpu_stats:
            th.cuda.synchronize()
            th.cuda.empty_cache()
            mem_gb = th.cuda.memory_reserved() / 1024**3
            print("[PDN(b)] Reserved GPU Mem (GB): ",mem_gb)
            mem_gb = th.cuda.memory_allocated() / 1024**3
            print("[PDN(b)] Allocated GPU Mem (GB): ",mem_gb)
            # print(th.cuda.memory_summary())

        noise = self.sep_net.sep_part2(inputs)
        deno,patches_w = self.run_batched_pdn_final(patches_n0,noise)

        return deno,patches_w

    def run_batched_pdn_final(self,patches_n0,noise):
        patches_dn = patches_n0[:, :, 0, :] - noise.squeeze(-2)
        patches_no_mean = patches_dn - patches_dn.mean(dim=-1, keepdim=True)
        patch_exp_weights = (patches_no_mean ** 2).mean(dim=-1, keepdim=True)
        patch_weights = th.exp(-self.beta.abs() * patch_exp_weights)
        return patches_dn,patch_weights

