
import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import LOSSES

@LOSSES.register_module
class NLLLoss(nn.Module):
    def __init__(self,
                 ratio = 0.2,
                 sample='std',
                 loss_weight = 1.0,
                 ignore_label = 255,
                 class_weight = None):
        super().__init__()
        assert sample in ['som', 'std']
        self.ratio = ratio
        self.sample = sample
        self.loss_weight = loss_weight
        self.ignore_label = ignore_label
        self.class_weight = class_weight

    def forward(self, score, target):
        if self.class_weight is not None:
            class_weight = score.new_tensor(self.class_weight)
        else:
            class_weight = None
        log_in = torch.log(score)
        losses = F.nll_loss(log_in, 
                            target, 
                            weight=class_weight,
                            ignore_index=self.ignore_label,
                            reduction='none')
        if self.sample == 'som':
            num_inst = losses.numel()
            num_hns = int(self.ratio * num_inst)
            top_loss, _ = losses.reshape(-1).topk(num_hns, -1)
            losses = top_loss[top_loss!=0]
        else:
            losses = losses[losses!=0]

        return self.loss_weight * losses.mean()

@LOSSES.register_module
class CrossEntropyLoss(nn.Module):

    def __init__(self,
                 ratio = 0.2,
                 sample='std',
                 loss_weight = 1.0,
                 ignore_label = 255,
                 class_weight = None):
        super().__init__()
        self.ratio = ratio
        self.sample = sample
        self.ignore_label = ignore_label
        self.loss_weight = loss_weight
        self.class_weight = class_weight
    def forward(self, score, target):

        if self.class_weight is not None:
            class_weight = score.new_tensor(self.class_weight)
        else:
            class_weight = None

        losses = F.cross_entropy(
            score,
            target,
            weight=class_weight,
            ignore_index=self.ignore_label,
            reduction='none').contiguous() 
        
        if self.sample == 'som':
            num_inst = losses.numel()
            
            num_hns = int(self.ratio * num_inst)
            top_loss, _ = losses.reshape(-1).topk(num_hns, -1)
            losses = top_loss[top_loss!=0]
        elif self.sample == 'ohem':
            mask = target.contiguous().view(-1) != self.ignore_label
            pred = F.softmax(score, dim=1)
            tmp_target = target.clone()  
            tmp_target[tmp_target == self.ignore_label] = 0
            pred = pred.gather(1, tmp_target.unsqueeze(1))
            pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
            threshold = pred.mean()
            losses = losses.view(-1)[mask][ind][pred < threshold]
        else:
            losses = losses[losses!=0]
        return self.loss_weight * losses.mean()
 