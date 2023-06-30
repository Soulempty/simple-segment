import os
import cv2
import torch
import numpy as np
import pandas as pd
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PALETTE = [ 
        (0,0,0), (0,0,255), (156,102,102), (0, 0, 142), (0, 0, 230), (106, 0, 228),
        (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
        (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
        (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
        (0, 82, 0)]

def get_ext(path):
    if os.path.isdir(path):
        for f in os.scandir(path):
            return get_ext(f.path)
    else:
        return os.path.splitext(path)[1]

def pre_to_img(img,mask,alpha=0.7):
    img = img.copy()
    ids = np.unique(mask)
    for id_ in ids:
        if id_ == 0:
            continue
        img[mask==id_] = np.array([PALETTE[id_]])*alpha + img[mask==id_]*(1-alpha) 
    return img

def compare(label_path,pred_path_1,pred_path_2,image_path,save_path,class_names):
    result_1 = []
    result_2 = []
    num_classes = len(class_names)
    img_sp = os.path.join(save_path,'image')
    os.makedirs(img_sp,exist_ok=True)
    for f in os.scandir(pred_path_1):
        path = f.path
        filename = f.name 
        basename = os.path.splitext(filename)[0]
        image_ext = get_ext(image_path)
        image = cv2.imread(os.path.join(image_path,basename+image_ext))
        label = cv2.imread(os.path.join(label_path,filename),0)
        pred_label_1 = cv2.imread(path,0) 
        pred_label_2 = cv2.imread(os.path.join(pred_path_2,filename),0)

        result_1.append(intersect_and_union(pred_label_1,label,num_classes))
        result_2.append(intersect_and_union(pred_label_2,label,num_classes))
        m1 = pre_to_img(image,label,alpha=0.7)
        m2 = pre_to_img(image,pred_label_1,alpha=0.7)
        m3 = pre_to_img(image,pred_label_2,alpha=0.7)
        result = np.concatenate([m1,m2,m3],1)
        cv2.imwrite(os.path.join(img_sp,filename),result)
    csv_path = os.path.join(save_path,'metrics.csv')
    if os.path.exists(csv_path):
        os.remove(csv_path)
    evaluate(result_1,result_2,class_names,csv_path)
    


def evaluate(result_1,result_2,class_names,csv_path):
    result_1 = tuple(zip(*result_1))
    result_1_intersect = sum(result_1[0])
    result_1_union = sum(result_1[1])
    result_1_pred_label = sum(result_1[2])
    result_1_label = sum(result_1[3])
 
    result_2 = tuple(zip(*result_2))
    result_2_intersect = sum(result_2[0])
    result_2_union = sum(result_2[1])
    result_2_pred_label = sum(result_2[2])
    result_2_label = sum(result_2[3])


    headers = []
    data = []
    metrics = []
    avg = ['average']
    headers.append('Class')
    headers.append('Pred_1 IoU')
    headers.append('Pred_2 IoU')
    iou_1 = result_1_intersect / (result_1_union+1)
    iou_2 = result_2_intersect / (result_2_union+1)
    data.append(list(iou_1))
    data.append(list(iou_2))
    avg.append(iou_1.mean())
    avg.append(iou_2.mean())
    
    headers.append('Pred_1 Recall')
    headers.append('Pred_2 Recall')
    acc_1 = result_1_intersect / (result_1_label+1)
    acc_2 = result_2_intersect / (result_2_label+1)
    data.append(list(acc_1))
    data.append(list(acc_2))
    avg.append(acc_1.mean())
    avg.append(acc_2.mean())

    headers.append('Pred_1 Precision')
    headers.append('Pred_2 Precision')
    precision_1 = result_1_intersect / (result_1_pred_label+1)
    precision_2 = result_2_intersect / (result_2_pred_label+1)
    data.append(list(precision_1))
    data.append(list(precision_2))
    avg.append(precision_1.mean())
    avg.append(precision_2.mean())
    
    for i,line in enumerate(zip(*data)):
        metrics.append([class_names[i],]+list(line)) 
    metrics.append(avg)
    df = pd.DataFrame(metrics,columns=headers)
    df.to_csv(csv_path,sep=',',index=False,header=True) 
    print(df)


def intersect_and_union(pred_label, label, num_classes, ignore_index=255):
    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]
    intersect = pred_label[pred_label == label]
    area_intersect = np.histogram(intersect, bins=num_classes, range=(0,num_classes-1))[0]
    area_pred_label = np.histogram(pred_label, bins=num_classes, range=(0,num_classes-1))[0]
    area_label = np.histogram(label, bins=num_classes, range=(0,num_classes-1))[0]
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label

if __name__ =="__main__":  
    class_names = ["background","Biaoji","Bianyuanbaidian","Huahen","Lvpao","Bengbian","Baidian","Heidian","Baiban","Xianyichang",
                "Yiwu","Zangwu","Lvbuqi","Duankai","Huashang","Cashang","Huanqie","Kailie","Xianyichang2","Lvbuping"]
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('--label_path',default='../../../dataset/wafer/labels')
    parser.add_argument('--pred_path_1',default='work_dirs/result')
    parser.add_argument('--pred_path_2',default='../../../dataset/wafer/mask')
    parser.add_argument('--image_path',default='../../../dataset/wafer/images')
    parser.add_argument('--save_path',default='../../../dataset/wafer/compare')
    args = parser.parse_args()

    compare(args.label_path,
            args.pred_path_1,
            args.pred_path_2,
            args.image_path,
            args.save_path,
            class_names)
   