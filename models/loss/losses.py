import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.registry import LOSSES

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
                 ratio = 0.1,
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
        print("class_weight:",self.class_weight)
    
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
            return self.loss_weight*self.som_loss(losses)
        elif self.sample == 'ohem':
            return self.loss_weight*self.ohem_loss(losses,target,score)
        elif self.sample == 'strap1':
            return self.loss_weight*self.strap1_loss(losses,target,score)
        elif self.sample == 'strap2':
            return self.loss_weight*self.strap2_loss(losses,target,score)
        elif self.sample == 'strap3':
            return self.loss_weight*self.strap3_loss(losses,target,score)
        elif self.sample == 'instance':
            return self.loss_weight*self.instance_loss(losses,target,score)
        else:
            return self.loss_weight * losses[losses!=0].mean()
    
    def som_loss(self,losses):
        num_inst = losses.numel()    
        num_hns = int(self.ratio * num_inst)
        top_loss, _ = losses.reshape(-1).topk(num_hns, -1)
        losses = top_loss[top_loss!=0]
        return losses.mean()
    
    def ohem_loss(self,losses,target,score):
        mask = target.contiguous().view(-1) != self.ignore_label
        pred = F.softmax(score, dim=1)
        tmp_target = target.clone()  
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
        threshold = pred.mean()
        losses = losses.view(-1)[mask][ind][pred < threshold]
        return losses.mean()
    
    def strap1_loss(self,losses,target,score):
        mask = target.contiguous().view(-1) != self.ignore_label
        tmp_target = target.clone()  
        N,H,W = tmp_target.size()
        tmp_target[tmp_target == self.ignore_label] = 0
        num_hns = int(min((tmp_target>0).sum().item(),H*W*N))
        if num_hns == 0:
            return self.ohem_loss(losses,target,score)
        
        top_loss, _ = losses.reshape(-1)[mask].topk(num_hns, -1)
        losses = top_loss[top_loss!=0]
        return losses.mean()
    
    def strap3_loss(self,losses,target,score):
        tmp_target = target.clone()  
        N,H,W = tmp_target.size()
        tmp_target[tmp_target == self.ignore_label] = 0
        if int((tmp_target>0).sum().item()) == 0:
            return self.ohem_loss(losses,target,score)
        
        loss = []
        for i in range(N):
            mask = target[i].contiguous().view(-1) != self.ignore_label
            num_hns = int(min((tmp_target[i]>0).sum().item(),H*W))
            top_loss, _ = losses[i].reshape(-1)[mask].topk(num_hns, -1)
            loss.append(top_loss)
        top_loss = torch.cat(loss,0)
        losses = top_loss[top_loss!=0]
        return losses.mean()
    
    def instance_loss(self,losses,target,score):
        label = target.clone()  
        label[label==self.ignore_label] = 0
        N,H,W = label.size()
        if (label>0).sum() == 0:
            return self.ohem_loss(losses,target,score)
 
        mask = label.new_zeros((N,H,W))
        
        for i in range(N):
            gt = label[i].cpu().numpy().astype(np.uint8)
            cls_ids = np.unique(gt)
            std_loss = 0
            for cls_id in cls_ids[1:]:
                m = np.zeros(gt.shape,gt.dtype)
                m[gt==cls_id] = 1
                contours,_ = cv2.findContours(m,mode=cv2.RETR_LIST,method=cv2.CHAIN_APPROX_SIMPLE)                
                for contour in contours:
                    x,y,w,h = cv2.boundingRect(contour)
                    xmin = max(0,x-10)
                    ymin = max(0,y-10)
                    xmax = min(W,x+w+10)
                    ymax = min(H,y+h+10)
                    mean_loss = losses[i,ymin:ymax,xmin:xmax].mean()
                    if mean_loss>std_loss:
                        std_loss = mean_loss
            mask[i][losses[i]>=std_loss] = 1

        losses = losses.view(-1)*mask.view(-1)
        losses = losses[losses!=0]
        return losses.mean()
    
    def strap2_loss(self,losses,target,score):
        label = target.clone()  
        label[label==self.ignore_label] = 0
        label[label>0] = 1
        N,H,W = label.size()
        global_loss = self.ohem_loss(losses,target,score)
        local_loss = 0
        for i in range(N):
            gt = label[i].cpu().numpy().astype(np.uint8)
            cls_ids = np.unique(gt)
            instance_loss = 0
            for cls_id in cls_ids[1:]:
                m = np.zeros(gt.shape,gt.dtype)
                m[gt==cls_id] = 1
                contours,_ = cv2.findContours(m,mode=cv2.RETR_LIST,method=cv2.CHAIN_APPROX_SIMPLE)                
                for contour in contours:
                    x,y,w,h = cv2.boundingRect(contour)
                    instance_loss = max(losses[i][y:y+h,x:x+w].mean(),instance_loss)
            local_loss += instance_loss
        
        losses = global_loss + local_loss/N
        return losses


@LOSSES.register_module
class BlobLoss(nn.Module):
    '''
    The original article refers to 
    Kofler, Florian and Shit. "blob loss: instance imbalance aware loss functions for semantic segmentation"
     (https://arxiv.org/pdf/2205.08209.pdf)
    '''
    def __init__(self,fp_ratio=0.1,
                 loss_weight = 1.0,
                 ignore_label = 255,
                 class_weight = None):
        super().__init__()
        self.fp_ratio = fp_ratio
        self.loss_weight = loss_weight
        self.ignore_label = ignore_label
        self.class_weight = class_weight

    def forward(self, score, target):
        '''
        score:  [N,NUM_CLASSES,H,W]
        target: [N,H,W]
        '''       
        N,C,H,W = score.size()
        label = target.clone()  
        label[label==self.ignore_label] = 0
        valid = target.contiguous() != self.ignore_label 
              
        class_weight = score.new_tensor([1]*C)
        if self.class_weight is not None:
            class_weight = score.new_tensor(self.class_weight)
 
        pred_score = F.softmax(score,dim=1)    
        pred_mask = torch.argmax(pred_score,dim=1)      # N, H, W
        pred = pred_score.gather(1, label.unsqueeze(1)).squeeze(1) # N,H,W
        log_ = -torch.log(pred)
        
        pred, ind = pred.contiguous().view(-1, )[valid.view(-1)].contiguous().sort()
        threshold = pred.mean()
        ohem_loss = log_.view(-1)[valid.view(-1)][ind][pred < threshold].mean()

        blob_loss = 0
        for i in range(N):
            valid_ = valid[i].view(-1)   
            gt = label[i].cpu().numpy().astype(np.uint8)
            cls_ids = np.unique(gt)
            instance_loss = 0
            fp_loss = 0
            for cls_id in cls_ids[1:]:  
                m = np.zeros(gt.shape,gt.dtype)
                m[gt==cls_id] = 1
                contours,_ = cv2.findContours(m,mode=cv2.RETR_LIST,method=cv2.CHAIN_APPROX_SIMPLE)     
                max_loss = 0
                for contour in contours:
                    mask = np.zeros(m.shape,m.dtype)
                    cv2.fillPoly(mask,[contour],1) 
                    mask = label.new_tensor(mask)
                    loss = (log_[i]*mask).reshape(-1)[valid_]
                    max_loss =  max(loss[loss!=0].mean(),max_loss)
                    
                roi = label.new_zeros((H,W))
                roi[pred_mask[i]==cls_id] = 1
                roi[label[i]==cls_id] = 0
                if roi.sum() > 0:  
                    loss_ = (log_[i]*roi).reshape(-1)[valid_]
                    fp_loss += loss_[loss_!=0].mean()*class_weight[0]
                instance_loss += max_loss * class_weight[cls_id]
            
            blob_loss += (instance_loss + fp_loss*self.fp_ratio) / max(len(cls_ids)-1,1)
        losses = ohem_loss + blob_loss / N
        return self.loss_weight*losses

