import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models.utils.arch_util import DWConv, LayerNorm, Itv_concat, SAM, ResidualBlockNoBN, kernel2d_conv, conv_block
from models.utils.transformerBLC_util import Standard_Mlp
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import logging

logger = logging.getLogger('base')


class FeedForward(nn.Module):
    """
    x: B,C,H,W
    return: B,C,H,W
    process: 1x1 conv + 3x3 dwconv + gate + 1x1 conv
    Adopted: Restormer —— Gated-Dconv Feed-Forward Network (GDFN)
    """
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class DW_Mlp(nn.Module):
    """
    x: B,hw,C
    return: B,hw,C
    process: mlp + 3x3 dwconv + gelu(drop) + mlp(drop)
    Adopted: Transweather
    """
    def __init__(self, dim, ffn_expansion_factor, bias, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.fc1 = nn.Conv2d(dim, hidden_features, 1, 1, 0, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, dim, 1, 1, 0, bias=bias)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class Key_Attention(nn.Module):
    """
    x: B,C,H,W   key: B,L,C
    return: B,C,H,W
    process: 1x1 qkv_conv + 3x3 dwconv + normalize(hw) + cxc attention(temperature) -> value + 1x1 conv
    Adopted: Restormer —— Multi-DConv Head Transposed Self-Attention (MDTA)
    ps: attention(scale) == normalize + attention
    """
    def __init__(self, dim, dimkey, num_heads, bias,qk_scale=None,attn_drop=0.,proj_drop=0.):
        super(Key_Attention, self).__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or dim ** -0.5
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        )
        self.kv = nn.Linear(dimkey, dim*2)

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, key):
        b, c, h, w = x.shape

        q = self.q(x)
        kv = self.kv(key)
        k, v = kv.chunk(2, dim=2)

        q = rearrange(q, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        k = rearrange(k, 'b k (head c) -> b head k c', head=self.num_heads)
        v = rearrange(v, 'b k (head c) -> b head k c', head=self.num_heads)


        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v)

        out = rearrange(out, 'b head (h w) c -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        out = self.proj_drop(out)
        return out

## Attention modules
class Key_TransformerBlock(nn.Module):
    """
    x: B,C,H,W   key: B,K,C
    return: B,C,H,W
    process: MDTA + GDFN
    params: dim, num_heads, ffn_expansion_factor, bias:True/false LayerNorm_type: BiasFree/WithBias
    Adopted:: Restormer
    """
    def __init__(self, dim, dimkey, num_heads, ffn_expansion_factor, bias, LayerNorm_type, principle=True, sam=False, ops_type=4, pred=False):
        super(Key_TransformerBlock, self).__init__()

        self.normkey = nn.LayerNorm(dimkey, elementwise_affine=False)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Key_Attention(dim, dimkey, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = DW_Mlp(dim, ffn_expansion_factor, bias)
        self.sam = sam
        self.principle = principle
        if principle:
            self.principle = PrincipleNet(dim=dim, ops_type=ops_type,pred=pred)
        if sam:
            self.SAM = SAM(n_feat=dim, kernel_size=1, bias=bias)

    def forward(self, im_degra, key, resize_img=None,degra_type=None):
        if degra_type is None:
            weight = self.principle.pred(im_degra).squeeze()
            dynamic_S = weight @ key[1]
            if len(dynamic_S.shape)==1:
                dynamic_S = dynamic_S[None,:]
            dynamic_S = torch.stack(list(map(lambda x: torch.diag(x), dynamic_S)))
            key = key[0]@dynamic_S@key[2]

        if self.sam:
            degra_map, img = self.SAM(im_degra,resize_img)
            degra_map = self.attn(self.norm1(degra_map), self.normkey(key))
        else:
            degra_map = self.attn(self.norm1(im_degra), self.normkey(key))

        if self.principle:
            im_degra, pred = self.principle(im_degra,degra_map,degra_type=degra_type)
        else:
            im_degra = im_degra - degra_map*im_degra 

        im_degra = im_degra + self.ffn(self.norm2(im_degra))

        if self.sam:
            return im_degra, img, pred
        else:
            return im_degra, None, pred


class PrincipleNet(nn.Module):
    def __init__(self,dim,pred=False,ops_type=6):
        super(PrincipleNet,self).__init__()
        self.mlp_img = nn.Conv2d(dim,dim,1,1,0)
        self.mlp_degra = nn.Conv2d(dim,dim,1,1,0)
        self.convolve = Convolve_route(dim=dim)
        self.addition = Addition_route(dim=dim)
        self.point = Point_route(dim=dim)
        self.identity = Identity_route()
        self.de_dict = {'denoise': self.addition, 'denoise_15': self.addition, 'denoise_25': self.addition, 'denoise_50': self.addition, 'derain': self.addition, 'dehaze': self.point, 'deblur': self.convolve, 'delowlight': self.point, 'clean': self.identity}
        self.flag = pred
        if pred:
            self.pred = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim,dim,1,1,0),
                nn.Conv2d(dim,ops_type,1,1,0),
                nn.Softmax(dim=1)
            )

    def forward(self,img,degra_map,degra_type=None):
        b,c,h,w = img.shape
        pred = self.pred(img)    #    B,C,H,W -> B,K,1,1

        img = self.mlp_img(img)
        degra_map = self.mlp_degra(degra_map)
        degra_map = Itv_concat(img,degra_map)
        if degra_type is not None:
            # stage 1 training
            fn = self.de_dict[degra_type]
            out = fn(img,degra_map)
            return out, pred
        else:
            # stage 2 training
            out2 = self.addition(img,degra_map)
            out3 = self.convolve(img,degra_map)
            out1 = self.point(img,degra_map)

            weight_point = pred[:,2,:,:]  + pred[:,4,:,:] 
            weight_addition = pred[:,0,:,:] + pred[:,1,:,:]
            weight_convolove = pred[:,3,:,:]

            out = weight_point.unsqueeze(-1)*out1 + weight_addition.unsqueeze(-1)*out2 + weight_convolove.unsqueeze(-1)*out3
            return out, pred

class Identity_route(nn.Module):
    def __init__(self):
        super(Identity_route,self).__init__()
    def forward(self,img_degra,degra_map):
        return img_degra

class Convolve_route(nn.Module):
    def __init__(self,dim,kpn_sz=5):
        super(Convolve_route,self).__init__()
        self.kpn_sz = kpn_sz
        self.convolve = nn.Sequential(
            conv_block(dim*2,dim,kernel_size=3),
            ResidualBlockNoBN(num_feat=dim),
            ResidualBlockNoBN(num_feat=dim),
            conv_block(dim,dim*(kpn_sz ** 2),kernel_size=1)
        )
    def forward(self,img_degra,degra_map):
        blur_kernel = self.convolve(degra_map)
        img_degra = kernel2d_conv(img_degra,blur_kernel,self.kpn_sz)
        return img_degra

class Addition_route(nn.Module):
    def __init__(self,dim):
        super(Addition_route,self).__init__()
        self.add = nn.Sequential(
            conv_block(dim*2,dim,kernel_size=3),
            ResidualBlockNoBN(num_feat=dim),
            ResidualBlockNoBN(num_feat=dim)
        ) 
    def forward(self,img_degra,degra_map):
        degradation = self.add(degra_map)
        img_degra += degradation
        return img_degra 

class Point_route(nn.Module):
    def __init__(self,dim):
        super(Point_route,self).__init__()
        self.weight = nn.Sequential(
            conv_block(2*dim,2*dim,kernel_size=3),
            ResidualBlockNoBN(num_feat=2*dim),
            ResidualBlockNoBN(num_feat=2*dim)
        )
        self.dim = dim
    def forward(self,img_degra,degra_map):
        weight = self.weight(degra_map)
        gamma, beta = weight.chunk(2,dim=1)
        img_degra = gamma*img_degra + beta
        return img_degra


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class RandomMLP_Mixer(nn.Module):
    """
    pca_token: full_cat tokens after pca -> compact token  B,K,C
    """
    def __init__(self,dim,num_degra,keep_degra,ffn_expansion_factor=2.66):
        super(RandomMLP_Mixer,self).__init__()
        self.mlp_channel = PreNormResidual(dim=dim, fn=Standard_Mlp(in_features=dim,hidden_features=int(dim * ffn_expansion_factor)))
        self.mlp_token = nn.Linear(num_degra,keep_degra)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self,pca_token):
        pca_token = pca_token.transpose(1,2)
        pca_token = self.mlp_token(pca_token)
        pca_token = pca_token.transpose(1,2)
        pca_token = self.mlp_channel(pca_token)
        return pca_token

class PI_MLP_Mixer(nn.Module):
    """
    pca_token: full_cat tokens after pca -> compact token  B,K,C
    """
    def __init__(self,dim,num_degra,keep_degra,ffn_expansion_factor=2.66,init='pca'):
        super(PI_MLP_Mixer,self).__init__()
        self.keep_degra = keep_degra
        self.init=init
        self.convU = nn.Conv2d(num_degra,int(num_degra//5),1,1,0)
        self.convV = nn.Conv2d(num_degra,int(num_degra//5),1,1,0)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
                
    def forward(self,U,V,B):
        U = self.convU(U)
        V = self.convV(V)
        U = U.squeeze().transpose(0,1)[None,:,:].expand(B,-1,-1)
        V = V.squeeze()[None,:,:].expand(B,-1,-1)
        return U,V
    
def process_USV(pca_token):
    U,S,V = torch.svd(pca_token)
    U = U.transpose(1,2)
    U = rearrange(U, 'n k c -> (n k) c')
    U = U[None,:,:,None]     # 1 nk, k, 1 -> 1 k, k, 1
    V = V.transpose(1,2)
    V =rearrange(V, 'n k c -> (n k) c')
    V = V[None,:,:,None]     # 1 nk, c, 1 -> 1 k, c, 1
    return U,S,V


def pca_init(x,keep_dim):
    """
    x: K,C
    """
    n = x.shape[0]
    mean = torch.mean(x,axis=0)
    x = x - mean
    covariance_matrix = 1 / n * torch.matmul(x, x.T)
    U,S,V = torch.svd(covariance_matrix)
    proj_matrix = U[:keep_dim,:]
    return proj_matrix

def svd_init(x,keep_dim):
    """
    x: K,C
    """
    U,S,V = torch.svd(x)
    proj_matrix = U[:keep_dim,:]
    return proj_matrix









