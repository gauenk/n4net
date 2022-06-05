
# -- linalg --
import torch as th
from einops import rearrange,repeat

# -- diff. non-local search --
import dnls

# -- separate class and logic --
from n4net.utils import clean_code
__methods__ = [] # self is a DataStore
register_method = clean_code.register_method(__methods__)

# -- helper imports --
from n4net.utils.inds import get_3d_inds
from .misc import get_image_params

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Run Nearest Neighbors Search
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

@register_method
def run_nn0(self,image_n,queryInds,scatter_nl,
            srch_img=None,flows=None,train=False,ws=29,wt=0,
            neigh_pad = 14):

    #
    # -- Our Search --
    #

    # -- pad & unpack --
    ps = self.patch_w
    device = image_n.device
    patch_numel = (self.patch_w ** 2) * image_n.shape[1]

    # -- prepeare image --
    image_n0 = self.pad_crop0(image_n, self.pad_offs, train)
    image_n0 = image_n0.contiguous()

    # -- params --
    params = get_image_params(image_n0, self.patch_w, neigh_pad)

    # -- get search image --
    if not(srch_img is None):
        img_nn0 = self.pad_crop0(image_n, self.pad_offs, train)
        img_nn0 = self.rgb2gray(img_nn0)
    elif self.arch_opt.rgb:
        img_nn0 = self.rgb2gray(image_n0)
    else:
        img_nn0 = image_n0
    img_nn0 = img_nn0.detach()

    # -- add padding --
    t,c,h0,w0 = image_n0.shape
    hp,wp = params['patches_h'],params['patches_w']
    sh,sw = (h0 - hp)//2,(w0 - wp)//2
    queryInds[...,1] += sh
    queryInds[...,2] += sw

    # -- search --
    print(queryInds[:3])
    print(queryInds[-3:])
    print("image_n0.shape: ",image_n0.shape)
    print("img_nn0.shape: ",img_nn0.shape)
    k,ps,pt,chnls = 14,self.patch_w,1,1
    nlDists,nlInds = dnls.simple.search.run(img_nn0,queryInds,flows,
                                            k,ps,pt,ws,wt,chnls)
    th.cuda.synchronize()
    # -- remove padding --
    queryInds[...,1] -= sh
    queryInds[...,2] -= sw

    #
    # -- Scatter Section --
    #

    # -- indexing patches --
    t,c,h,w = image_n0.shape
    # patches = dnls.simple.scatter.run(image_n0,nlInds,self.patch_w)
    patches = scatter_nl(image_n0,nlInds)
    ishape = 'p k 1 c h w -> p k (c h w)'
    patches = rearrange(patches,ishape)

    # -- append anchor patch spatial variance --
    d = patches.shape[-1]
    nlDists[:,:-1] = nlDists[...,1:]
    nlDists[:,-1] = patches[:,0,:].std(dim=-1).pow(2)*d # patch var

    # -- remove padding --
    nlInds[...,1] -= sh
    nlInds[...,2] -= sw

    return patches,nlDists,nlInds,params

@register_method
def run_nn1(self,image_n,queryInds,scatter_nl,
            srch_img=None,flows=None,train=False,ws=29,wt=0,
            neigh_pad = 14):

    # -- unpack --
    t = image_n.shape[0]
    ps = self.patch_w
    device = image_n.device
    patch_numel = (self.patch_w ** 2) * image_n.shape[1]

    # -- pad & unpack --
    image_n1 = self.prepare_image_n1(image_n,train)
    params = get_image_params(image_n1, 2*self.patch_w-1, 2*neigh_pad)

    #
    #  -- DNLS Search --
    #

    # -- get search image --
    if not(srch_img is None):
        img_nn1 = self.prepare_image_n1(srch_img,train)
        img_nn1 = self.rgb2gray(img_nn1)
    elif self.arch_opt.rgb:
        img_nn1 = self.rgb2gray(image_n1)
    else:
        img_nn1 = image_n1
    img_nn1 = img_nn1.detach()

    # -- inds offsets --
    t,c,h0,w0 = image_n1.shape
    hp,wp = params['patches_h'],params['patches_w']
    sh,sw = (h0 - hp)//2,(w0 - wp)//2
    queryInds[...,1] += sh
    queryInds[...,2] += sw

    # -- exec search --
    # print(queryInds[:3])
    # print(queryInds[-3:])
    # print("image_n1.shape: ",image_n1.shape)
    # print("img_nn1.shape: ",img_nn1.shape)
    th.cuda.synchronize()
    k,pt,chnls = 14,1,1
    nlDists,nlInds = dnls.simple.search.run(img_nn1,queryInds,flows,
                                            k,ps,pt,ws,wt,chnls,
                                            stride=2,dilation=2)
    th.cuda.synchronize()

    # -- remove padding --
    queryInds[...,1] -= sh
    queryInds[...,2] -= sw

    #
    # -- Scatter Section --
    #

    # -- dnls --
    patches = scatter_nl(image_n1,nlInds)

    #
    # -- Final Formatting --
    #

    # - reshape --
    # hp,wp = params['patches_h'],params['patches_w']
    # nlDists = rearrange(nlDists,'(t h w) k -> t h w k',t=t,h=hp)
    # nlInds = rearrange(nlInds,'(t h w) k tr -> t h w k tr',t=t,h=hp)
    # ishape = '(t ih iw) k 1 c h w -> t ih iw k (c h w)'
    # patches = rearrange(patches,ishape,ih=hp,iw=wp)
    ishape = 'p k 1 c h w -> p k (c h w)'
    patches = rearrange(patches,ishape)

    # -- patch variance --
    d = patches.shape[-1]
    nlDists[:,:-1] = nlDists[...,1:]
    nlDists[:,-1] = patches[:,0,:].std(-1)**2*d # patch_var

    # -- centering inds --
    t,c,h1,w1 = image_n1.shape
    sh,sw = (h1 - hp)//2,(w1 - wp)//2
    nlInds[...,1] -= sh
    nlInds[...,2] -= sw

    return patches,nlDists,nlInds,params
