import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
import math
from models.utils.arch_util import DWConv, LayerNorm
from einops import rearrange

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed_Keep(nn.Module):
    """
    x: B,C1,H,W
    return: B,C2,H,W
    process: 单conv层
    Adopted: Restormer
    """
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed_Keep, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class OverlapPatchEmbed_Keep2(nn.Module):
    """
    x: B,C1,H,W
    return: B,C2,H,W
    process: 单conv层
    Adopted: Restormer
    """
    def __init__(self, in_c=3, embed_dim=48, patch_size=4, bias=False):
        super(OverlapPatchEmbed_Keep2, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class PatchEmbedding(nn.Module):
    """
    x: B,C1,H,W
    return: B,C2,H,W
    process: 单conv层
    Adopted: Restormer
    """
    def __init__(self, in_c=3, embed_dim=48, patch_size=4, bias=False):
        super(OverlapPatchEmbed_Keep2, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        nn.PixelUnshuffle(2)

    def forward(self, x):
        x = self.proj(x)

        return x

class OverlapPatchEmbed_Stride(nn.Module):
    """ Image to Patch Embedding
        x: B,C1,H,W
        return: B,h*w,C2
        process: conv + layernorm (C2)   分辨率降四倍
        Adopted: Transweather
    """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

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
        # pdb.set_trace()
        x = self.proj(x)
        b,c,h,w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x.view(b,c,h,w)

class NonOverlapPatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    x: B,C,H,W
    return: B,C,h,w (/patch_size)
    process: conv(patch_slide) + norm
    Adopted: MAE
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        b, c, h, w = x.shape
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x.view(b,c,h,w)

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    """
    x: B,C,H,W
    return: B,2C,H/2,W/2
    process: conv + pixel-unshuffle (降C,分辨率补C)
    Adopted: Restormer
    """
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))      # (1,4,16,16) -> (1,16,8,8)

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """
    x: B,C,H,W
    return: B,C/2,2H,2W
    process: conv + pixel-shuffle (升C, C补分辨率)
    Adopted: Restormer
    """
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))        # (1,4,16,16) -> (1,1,32,32)

    def forward(self, x):
            return self.body(x)


class Upsample_BCHW(nn.Module):
    """
    x: B,C,H,W
    return: B,2C,h/2,w/2
    process: conv + pixel-shuffle (升C,C补分辨率)  分辨率double,channel减半
    Adopted: Restormer
    """
    def __init__(self, in_feat, type='pixelshuffle'):
        super(Upsample_BCHW, self).__init__()

        if type=='pixelshuffle':
            self.body = nn.Sequential(nn.Conv2d(in_feat, in_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.PixelShuffle(2))
        elif type=='ConvTranspose':
            self.body = nn.ConvTranspose2d(in_feat, in_feat//2, kernel_size=4, stride=2, padding=1)
        elif type=='Bilinear':
            self.body = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                      nn.Conv2d(in_feat,in_feat//2,1,1,0))

    def forward(self, x):
        x = self.body(x)
        return x


##########################################################################
## MLP modules
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
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features,1,1,0)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features,1,1,0)
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

class Standard_Mlp(nn.Module):
    """
    input: B,hw,C
    return: B,hw,C
    process: mlp(up) + gelu(drop) + mlp(down)(drop)
    Adopted: Swin-transformer
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features,1,1,0)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features,1,1,0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

##########################################################################
## Attention modules
class MDTA_Attention(nn.Module):
    """
    x: B,C,H,W
    return: B,C,H,W
    process: 1x1 qkv_conv + 3x3 dwconv + normalize(hw) + cxc attention(temperature) -> value + 1x1 conv
    Adopted: Restormer —— Multi-DConv Head Transposed Self-Attention (MDTA)
    ps: attention(scale) == normalize + attention
    """
    def __init__(self, dim, num_heads, bias):
        super(MDTA_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # print("MDTAQ:",q.mean())
        # print("MDTAK:",k.mean())
        # print("MDTAV:",v.mean())

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # print("MDTA_attenbs:",attn.mean())
        attn = attn.softmax(dim=-1)
        # print("MDTA_attenas:",attn.mean())
        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # print("MDTA_Aoutput:",out.mean())

        out = self.project_out(out)
        return out


class Standard_Attention(nn.Module):
    """
    x: B,hw,C
    return: B,hw,C
    process: mlp_q + mlp_kv(optional sr) + HW x hw attention(scale)(drop_path) -> value + mlp(drop)
    Adopted: Transweather
    ps: kv_sr: window slide conv + layernorm(C) + mlp_kv
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Conv2d(dim, dim, 1,1,0, bias=qkv_bias)
        self.kv = nn.Conv2d(dim, dim*2, 1,1,0, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim,1,1,0)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = LayerNorm(dim,'WithBias')

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

        b,c,h,w = x.shape
        q = self.q(x)  #.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = rearrange(q, 'b (head c) h w -> b head (h w) c', head=self.num_heads)

        if self.sr_ratio > 1:
            x = self.sr(x)
            x = self.norm(x)

        kv = self.kv(x)
        k, v = kv.chunk(2, dim=1)
        k = rearrange(k, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head (h w) c', head=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = rearrange(x, 'b head (h w) c -> b (head c) h w', head=self.num_heads, h=h, w=w)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Task_Attention(nn.Module):
    """
    x: B,hw,C
    return: B,hw,C
    process: mlp_q(task_query) + mlp_kv(optional sr) + interpolate(q) + HW x hw attention(scale)(drop_path) -> value + mlp(drop)
    Adopted: Transweather
    ps: interpolate(q): ensure output size
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Conv2d(dim, dim, 1,1,0, bias=qkv_bias)
        self.kv = nn.Conv2d(dim, dim*2, 1,1,0, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1,1,0)
        self.proj_drop = nn.Dropout(proj_drop)

        self.task_query = nn.Parameter(torch.randn(1, 48, dim))
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = LayerNorm(dim,'WithBias')

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

    def forward(self, x, H, W):

        b,n,c = x.shape
        task_q = self.task_query

        # This is because we fix the task parameters to be of a certain dimension, so with varying batch size, we just stack up the same queries to operate on the entire batch
        if b > 1:
            task_q = task_q.expand(b,-1,-1)

        q = self.q(task_q) 
        q = rearrange(q, 'b (head c) h w -> b head (h w) c', head=self.num_heads)

        if self.sr_ratio > 1:
            x = self.sr(x)
            x = self.norm(x)

        kv = self.kv(x)
        k, v = kv.chunk(2, dim=1)
        k = rearrange(k, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head (h w) c', head=self.num_heads)

        q = torch.nn.functional.interpolate(q, size=(v.shape[2], v.shape[3]))

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = rearrange(x, 'b head (h w) c -> b (head c) h w', head=self.num_heads, h=h, w=w)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


##########################################################################
## Transforemr block
class MDTA_TransformerBlock(nn.Module):
    """
    x: B,C,H,W
    return: B,C,H,W
    process: MDTA + GDFN
    params: dim, num_heads, ffn_expansion_factor, bias:True/false LayerNorm_type: BiasFree/WithBias
    Adopted:: Restormer
    """
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(MDTA_TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = MDTA_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class Standard_TransformerBlock(nn.Module):
    """
    x: B,hw,C
    return: B,hw,C
    process: Standard_Attention + mlp(optional:3x3 dwconv)
    params: dim, num_heads, mlp_ratio
    Adopted: Transweather (Standard_mlp -> mlp)
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, task=False,LayerNorm_type='WithBias'):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        if task:
            self.attn = Task_Attention(
                dim,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        else:
            self.attn = Standard_Attention(
                dim,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.mlp = Standard_Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

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

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Cross_TransformerBlock(nn.Module):
    """
    x: B,hw,C; B,HW,C
    return: B,hw,C
    process: Standard_Attention + mlp(3x3 dwconv)
    params: dim, num_heads, mlp_ratio
    Adopted: Transweather (Standard_mlp -> mlp)
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1,LayerNorm_type='WithBias'):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Query_Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.mlp = Standard_Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

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

    def forward(self, query, x):

        query = query + self.drop_path(self.attn(self.norm1(query), x))
        query = query + self.drop_path(self.mlp(self.norm2(query)))
        return query
