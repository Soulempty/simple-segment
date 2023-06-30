import torch
import random
import numpy as np
from threading import Thread
from tabulate import tabulate
from torch.utils.data import DataLoader
import sys

def async_(f):
  def wrapper(*args, **kwargs):
    thr = Thread(target=f, args=args, kwargs=kwargs)
    thr.start()
 
  return wrapper

def intersect_and_union(pred_label, label, num_classes, ignore_index=255):
    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]
    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(intersect.float(), bins=(num_classes), min=0,max=num_classes - 1).cpu()
    area_pred_label = torch.histc(pred_label.float(), bins=(num_classes), min=0,max=num_classes - 1).cpu()
    area_label = torch.histc(label.float(), bins=(num_classes), min=0,max=num_classes - 1).cpu()
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label

def evaluate(model,dataset,class_names,device=None,num_workers=1,metrics=['aAcc','IoU','Acc','Precision','Dice']):
    model.eval()
    loader = DataLoader(dataset,batch_size=1, shuffle=False, num_workers=num_workers)
    num_classes = len(class_names)
    results = []
    with torch.no_grad():
        for iter, data in enumerate(loader):
            images = data['img'].to(device)
            label = data['label'].to(device).squeeze()
            logit = model(images)[0]
            mask = torch.argmax(logit,dim=1)[0]
            results.append(intersect_and_union(mask,label,num_classes))
    
    results = tuple(zip(*results))
    total_area_intersect = sum(results[0])
    total_area_union = sum(results[1])
    total_area_pred_label = sum(results[2])
    total_area_label = sum(results[3])
    headers = []
    datas = []
    ret_metrics = []
    mean_iou = 0
    all_acc = 0
    cls_iou = None
    for metric in metrics:
        if metric == 'IoU':
            headers.append('mIoU')
            headers.append('IoU')
            iou = total_area_intersect / (total_area_union+1)
            cls_iou = iou
            mean_iou = iou[1:].mean()
            datas.append([mean_iou]*num_classes)
            datas.append(list(iou))
        elif metric == 'aAcc':
            headers.append('aAcc')
            all_acc = total_area_intersect[1:].sum() / total_area_label[1:].sum()
            datas.append([all_acc]*num_classes)
        elif metric == 'Acc':
            headers.append('Acc')
            acc = total_area_intersect / (total_area_label+1)
            datas.append(list(acc))
        elif metric == 'Precision':
            headers.append('Precision')
            precision = total_area_intersect / (total_area_pred_label+1)
            datas.append(list(precision))
        elif metric == 'Dice':
            headers.append('Dice')
            dice = 2 * total_area_intersect / (total_area_pred_label + total_area_label+1)
            datas.append(list(dice))

    for i,line in enumerate(zip(*datas)):
        ret_metrics.append([class_names[i],]+list(line))  

    print(tabulate(ret_metrics,headers))  

    return mean_iou, cls_iou

def set_seed(seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)


def cal_params(model): # M
    params = 1.0 * sum(param.numel() for param in model.parameters())/1000000
    return params

