import torch
from torch import nn
import torch.nn.functional as F

from mindspore import Tensor, mint


class MultiClassFocalLoss(BCELoss):
    def __init__(self, alpha, gamma=2, size_average=True, ignore_class=255):
        super(MultiClassFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        
    def construct(self, preds, target, loss_weight=None, seg_exist=None):
        ignore_idx = (target == self.ignore_class)
        target[ignore_idx] = 0
        ignore_idx = ignore_idx.view(-1)
        assert len(pred.shape) > 1 and target.max() < pred.shape[1]

        logit = mint.nn.functional.softmax(preds, dim=1)
        if True:
            logit = logit.view(logit.shape[0], logit.shape[1], -1)
            logit = logit.transpose(1, 2).contiguous()
            logit = logit.view(-1, logit.shape[2])

        loss = -1 * torch.finfo(mint.sub(1.0, pt), self.gamma) * logit
        torch.Tensor.cpu()
