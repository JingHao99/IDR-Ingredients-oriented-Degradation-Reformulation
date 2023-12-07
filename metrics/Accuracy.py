import torch
import torch.nn as nn

class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy,self).__init__()

    def forward(self,pred,label):
        """
        pred: one-hot  B,Nclass
        label: one-hot
        """
        pred = torch.argmax(pred, 1)
        label = torch.argmax(label, 1)
        accuracy = torch.eq(pred,label.squeeze(dim=-1)).float().mean()
        return accuracy