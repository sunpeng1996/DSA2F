import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn
class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        predict = F.softmax(predict, dim=1)[:,1,:,:].unsqueeze(1)
        target = target.unsqueeze(1).float()
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean() *256*256
        elif self.reduction == 'sum':
            return loss.sum()*256*256
        elif self.reduction == 'none':
            return loss*256*256
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))



def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()

    input = input.transpose(1, 2).transpose(2, 3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0] # 262144 #input = 2*256*256*2
    input = input.view(-1, c)
    mask = target >= 0
    target = target[mask]
    loss = F.cross_entropy(input, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss



def iou(pred, target, size_average = False):

    pred = F.softmax(pred, dim=1)
    IoU = 0.0
    Iand1 = torch.sum(target.float() * pred[:,1,:,:])
    Ior1 = torch.sum(target) + torch.sum(pred[:,1,:,:]) - Iand1
    IoU1 = (Iand1 + 1) / (Ior1 + 1)

    IoU = (1-IoU1)

    if size_average:
        IoU /= target.data.sum()
    return IoU * 256 * 256


# class CrossEntropyLabelSmooth(nn.Module):
#     """Cross entropy loss with label smoothing regularizer.
#     Reference:
#     Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
#     Equation: y = (1 - epsilon) * y + epsilon / K.
#     Args:
#         num_classes (int): number of classes.
#         epsilon (float): weight.
#     """

#     def __init__(self, num_classes=1000, epsilon=0.1):
#         super(CrossEntropyLabelSmooth, self).__init__()
#         self.num_classes = num_classes
#         self.epsilon = epsilon
#         self.logsoftmax = nn.LogSoftmax(dim=1)

#     def forward_v1(self, inputs, targets):
#         log_probs = self.logsoftmax(inputs)
#         targets = torch.zeros(log_probs.size(), device=targets.device).scatter_(1, targets.unsqueeze(1), 1)
#         targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
#         loss = (- targets * log_probs).mean(0).sum()
#         return loss
    
#     def forward_v2(self, inputs, targets):
#         probs = self.logsoftmax(inputs)
#         targets = torch.zeros(probs.size(), device=targets.device).scatter_(1, targets.unsqueeze(1), 1)
#         targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
#         loss = nn.KLDivLoss()(probs, targets)
#         return loss

#     def forward(self, inputs, targets):
#         """
#         Args:
#             inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
#             targets: ground truth labels with shape (num_classes)
#         """
#         return self.forward_v1(inputs, targets)