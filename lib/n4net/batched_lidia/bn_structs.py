
# -- linalg --
import torch as th

# -- neural network --
import torch.nn as nn

class VerHorBnRe(nn.Module):
    def __init__(self, ver_in, ver_out, hor_in, hor_out, bn, name=""):
        super(VerHorBnRe, self).__init__()
        self.ver_hor = VerHorMat(ver_in=ver_in, ver_out=ver_out,
                                 hor_in=hor_in, hor_out=hor_out)
        if bn: self.bn = nn.BatchNorm2d(hor_out)
        self.thr = nn.ReLU()
        self.name = name

    def forward(self, x):
        x = self.ver_hor(x)

        # -- main event --
        if hasattr(self, 'bn'):
            # -- testing --
            xt = x.transpose(-2, -3)
            # means = xt.mean((0,2,3))[None,:,None,None]
            # stds = xt.std((0,2,3))[None,:,None,None]
            # th.save(means,"means_%s" % self.name)
            # th.save(stds,"std_%s" % self.name)
            # print(self.name)
            means = th.load("means_%s" % self.name)
            stds = th.load("std_%s" % self.name)

            from easydict import EasyDict as edict
            from einops import rearrange

            params = edict()
            for key,val in self.bn.named_parameters():
                # print(key,val.shape)
                params[key] = rearrange(val,'k -> 1 k 1 1')
            # print(xt.shape)
            # -- exec bn --
            eps = 1e-8
            invsig = 1./th.pow(stds**2+eps,0.5)
            x = (xt - means) * invsig * params.weight + params.bias
            x = x.transpose(-2, -3)

            # -- bn --
            # x = self.bn(x.transpose(-2, -3)).transpose(-2, -3)

        x = self.thr(x)
        return x

class VerHorMat(nn.Module):
    def __init__(self, ver_in, ver_out, hor_in, hor_out):
        super(VerHorMat, self).__init__()
        self.ver = nn.Linear(in_features=ver_in, out_features=ver_out, bias=False)
        self.hor = nn.Linear(in_features=hor_in, out_features=hor_out, bias=False)
        self.b = nn.Parameter(th.empty(hor_out, ver_out, dtype=th.float32))
        nn.init.xavier_normal_(self.b, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x = self.ver(x)
        x = self.hor(x.transpose(-1, -2)).transpose(-1, -2)
        x = x + self.b
        return x

    def extra_repr(self):
        return 'b.shape=' + str(tuple(self.b.shape))

    def get_ver_out(self):
        return self.ver.out_features

    def get_hor_out(self):
        return self.hor.out_features

