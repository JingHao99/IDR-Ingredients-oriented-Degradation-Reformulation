import torch
from torchvision.transforms import Resize 
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
from models.utils.arch_util import LayerNorm
from models.utils.transformerBCHW_util import Downsample, Upsample, MDTA_TransformerBlock, OverlapPatchEmbed_Keep
from models.utils.module import Key_TransformerBlock, PI_MLP_Mixer, process_USV
import numpy as np

from einops import rearrange


##########################################################################
class IDR_restormer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 num_degra_queries = 24,
                 keep_degra = 48,
                 degra_type = 5,
                 sam = True,
                 ops_type = 5,
                 pred = True
                 ):
        super(IDR_restormer, self).__init__()

        self.de_dict = {'denoise': 0, 'denoise_15': 0, 'denoise_25': 0, 'denoise_50': 0, 'derain': 1, 'dehaze': 2, 'deblur': 3, 'delowlight': 4, 'clean': 5}

        self.patch_embed =OverlapPatchEmbed_Keep(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            MDTA_TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            MDTA_TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            MDTA_TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            MDTA_TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            MDTA_TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            MDTA_TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            MDTA_TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            MDTA_TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.degra_key = nn.Parameter(torch.randn(degra_type, num_degra_queries, int(dim * 2 ** 3)), requires_grad=True)
        self.dmixer = PI_MLP_Mixer(dim=int(dim * 2 ** 3),num_degra=num_degra_queries*degra_type,keep_degra=keep_degra,init='pca')
        self.kdp_level1 = Key_TransformerBlock(dim=dim, dimkey=int(dim * 2 ** 3), num_heads=heads[0], ffn_expansion_factor=2.66, bias=bias, LayerNorm_type=LayerNorm_type,principle=True, sam=sam, ops_type=ops_type,pred=pred)
        self.kdp_level2 = Key_TransformerBlock(dim=int(dim * 2 ** 1), dimkey=int(dim * 2 ** 3), num_heads=heads[1], ffn_expansion_factor=2.66, bias=bias, LayerNorm_type=LayerNorm_type,principle=True, sam=sam, ops_type=ops_type,pred=pred)
        self.kdp_level3 = Key_TransformerBlock(dim=int(dim * 2 ** 2), dimkey=int(dim * 2 ** 3), num_heads=heads[2], ffn_expansion_factor=2.66, bias=bias, LayerNorm_type=LayerNorm_type,principle=True, sam=sam, ops_type=ops_type,pred=pred)
        self.cri_pix = nn.L1Loss().cuda()



    def forward(self, inp_img, degra_type=None, gt=None, epoch=None):
        """
        only input_image is required during inference
        """
        flag=0
        batch_size,c,h,w = inp_img.shape
        if epoch and epoch <= 550:
            # stage 1 training - Task-oriented knowledge collection
            de_type = degra_type[0]
            degra_id = self.de_dict[de_type]
            degra_key = self.degra_key[degra_id,:,:].unsqueeze(0).expand(batch_size,-1,-1)
        else:
            # stage 2 training - Ingredients-oriented knowedge intergation
            if flag==0:
                U,S,V = process_USV(self.degra_key.detach())
                flag=1
            U,V = self.dmixer(U,V,batch_size)
            degra_key = [U,S,V]
            de_type = None


        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        torch_resize1 = Resize([out_enc_level1.shape[2],out_enc_level1.shape[3]])
        inp_img1 = torch_resize1(inp_img)
        out_enc_level1,output_img1,pred1 = self.kdp_level1(out_enc_level1,degra_key,inp_img1,degra_type=de_type)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        torch_resize2 = Resize([out_enc_level2.shape[2],out_enc_level2.shape[3]])
        inp_img2 = torch_resize2(inp_img)
        out_enc_level2,output_img2,pred2 = self.kdp_level2(out_enc_level2,degra_key,inp_img2,degra_type=de_type)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        torch_resize3 = Resize([out_enc_level3.shape[2],out_enc_level3.shape[3]])
        inp_img3 = torch_resize3(inp_img)
        out_enc_level3,output_img3,pred3 = self.kdp_level3(out_enc_level3,degra_key,inp_img3,degra_type=de_type)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        
        if gt is not None:
            gt_img1 = torch_resize1(gt)
            gt_img2 = torch_resize2(gt)
            gt_img3 = torch_resize3(gt)
            output_img = [output_img1,output_img2,output_img3] 
            gt_img = [gt_img1,gt_img2,gt_img3] 
            loss = np.sum([self.cri_pix(output_img[j],gt_img[j]) for j in range(len(output_img))])
            return [out_dec_level1,loss,pred1,pred2,pred3]
        else:
            return out_dec_level1

