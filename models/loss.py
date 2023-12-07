import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F



class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


class Vgg19(nn.Module):
    def __init__(self, id, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg = models.vgg19(pretrained=False)
        vgg.load_state_dict(torch.load('/gdata2/zhangjh/pretrain_models/vgg19-dcbb9e9d.pth'))
        vgg.eval()
        vgg_pretrained_features = vgg.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(3):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(3, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        self.id = id
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        if X.shape[1] == 3:
            h_relu1 = self.slice1(X)
            h_relu2 = self.slice2(h_relu1)
            h_relu3 = self.slice3(h_relu2)
            h_relu4 = self.slice4(h_relu3)
            h_relu5 = self.slice5(h_relu4)
            out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        elif X.shape[1] == 64:
            h_relu2 = self.slice2(X)
            h_relu3 = self.slice3(h_relu2)
            h_relu4 = self.slice4(h_relu3)
            h_relu5 = self.slice5(h_relu4)
            out = [h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self, id, gpu_id=0):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19(id).cuda(gpu_id)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self,x, y):
        while x.size()[3] > 4096:
            x, y = self.downsample(x), self.downsample(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        # loss = self.criterion(x_vgg, y_vgg.detach())
        if len(x_vgg) == 5:
            for i in range(len(x_vgg)):
                loss +=  self.weights[i]*self.criterion(x_vgg[i], y_vgg[i].detach())
        elif len(x_vgg) == 4:
            for i in range(len(x_vgg)):
                loss +=  self.weights[i+1]*self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

############################################################################################################################3

class WeightedTVLoss(nn.Module):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, reduction='mean'):
        super(WeightedTVLoss, self).__init__()
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        self.criterion = nn.L1Loss(reduction=reduction)

    def forward(self, pred):

        y_diff = self.criterion(pred[:, :, :-1, :], pred[:, :, 1:, :])
        x_diff = self.criterion(pred[:, :, :, :-1], pred[:, :, :, 1:])

        loss = x_diff + y_diff

        return loss


############################################################################################################################3
class EdgeLoss(nn.Module):

    def __init__(self,gpu_id):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda(gpu_id)
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss

############################################################################################################################3
class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class IMP_Loss(nn.Module):
    """
    attention_map: h,b,k
    """
    def __init__(self):
        super(IMP_Loss,self).__init__()
        self.criterion=nn.L1Loss()

    def forward(self,attention_map):
        imp = torch.sum(attention_map,axis=1,keepdim=False)
        loss = torch.mean((torch.std(imp,axis=1,keepdim=False)/torch.mean(imp,axis=1,keepdim=False))**2)
        return loss

# class SimDLoss(nn.Module):
#     """
#     attention_map: b,k
#     loss越小, 特异性越强,上限0.8
#     """
#     def __init__(self,gpu_id=0):
#         super(SimDLoss,self).__init__()
#     def forward(self,attention_map):
#         attention_map = F.normalize(attention_map, dim=1)
#         degra_matrix = attention_map @ attention_map.transpose(0,1)
#         degra_matrix = degra_matrix.softmax(1)
#         diag = (1-torch.eye(attention_map.shape[0])).cuda(gpu_id)
#         loss = torch.mean(torch.sum(degra_matrix*diag,axis=1))
#         return loss

class SimDLoss(nn.Module):
    """
    attention_map: h,b,k
    """
    def __init__(self,uniform=True,gpu_id=0,eps=1e-6):
        super(SimDLoss,self).__init__()
        self.uniform = uniform
        self.eps = eps
    def forward(self,attention_map):
        attention_map = F.normalize(attention_map, dim=2)
        degra_matrix = attention_map @ attention_map.transpose(1,2)
        # degra_matrix = degra_matrix.softmax(2)
        if self.uniform:
            degra_matrix = degra_matrix[:,-1,:]
            loss = torch.mean((torch.std(degra_matrix,axis=1)+self.eps)/torch.mean(degra_matrix,axis=1))
            # print(torch.mean(degra_matrix,axis=2))
        else:
            loss = torch.mean(torch.mean(degra_matrix,axis=2)/(torch.std(degra_matrix,axis=2)+self.eps))
            # print(torch.mean(degra_matrix,axis=2))
        return loss

############################################################################################################################################################
class OHCeLoss(nn.Module):
    def __init__(self):
        super(OHCeLoss,self).__init__()
    def forward(self,pred,onehot_label):
        pred = pred.squeeze()
        onehot_label = onehot_label.squeeze()
        N = pred.size(0)
        # log_prob = F.log_softmax(pred, dim=1)
        log_prob = torch.log(pred)
        loss = -torch.sum(log_prob * onehot_label) / N
        return loss


