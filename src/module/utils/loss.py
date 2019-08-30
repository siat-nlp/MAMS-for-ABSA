import torch
from torch import nn
import torch.nn.functional as F

class CapsuleLoss(nn.Module):

    def __init__(self, smooth=0.1, lamda=0.6):
        super(CapsuleLoss, self).__init__()
        self.smooth = smooth
        self.lamda = lamda

    def forward(self, input, target):
        one_hot = torch.zeros_like(input).to(input.device)
        one_hot = one_hot.scatter(1, target.unsqueeze(-1), 1)
        a = torch.max(torch.zeros_like(input).to(input.device), 1 - self.smooth - input)
        b = torch.max(torch.zeros_like(input).to(input.device), input - self.smooth)
        loss = one_hot * a * a + self.lamda * (1 - one_hot) * b * b
        loss = loss.sum(dim=1, keepdim=False)
        return loss.mean()

# CrossEntropyLoss for Label Smoothing Regularization
class CrossEntropyLoss_LSR(nn.Module):
    def __init__(self, para_LSR=0.2):
        super(CrossEntropyLoss_LSR, self).__init__()
        self.para_LSR = para_LSR
        self.logSoftmax = nn.LogSoftmax(dim=-1)

    def _toOneHot_smooth(self, label, batchsize, classes):
        prob = self.para_LSR * 1.0 / classes
        one_hot_label = torch.zeros(batchsize, classes) + prob
        for i in range(batchsize):
            index = label[i]
            one_hot_label[i, index] += (1.0 - self.para_LSR)
        return one_hot_label

    def forward(self, pre, label, size_average=True):
        b, c = pre.size()
        one_hot_label = self._toOneHot_smooth(label, b, c).to(pre.device)
        loss = torch.sum(-one_hot_label * self.logSoftmax(pre), dim=1)
        if size_average:
            return torch.mean(loss)
        else:
            return torch.sum(loss)

class SmoothCrossEntropy(nn.Module):

    def __init__(self, smooth=0.08):
        super(SmoothCrossEntropy, self).__init__()
        self.kldiv = nn.KLDivLoss()
        self.smooth = smooth

    def forward(self, input, target):
        one_hot = torch.zeros_like(input).to(input.device)
        one_hot = one_hot.scatter(1, target.unsqueeze(-1), 1)
        target = (1 - self.smooth) * one_hot + self.smooth / (input.size(1) - 1) * (1 - one_hot)
        # target = target + torch.rand_like(target).to(target.device) * 0.001
        input = input - input.max(dim=1, keepdim=True)[0]
        loss = -target * F.log_softmax(input, dim=-1)
        return loss.mean()