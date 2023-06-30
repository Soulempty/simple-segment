import os
import cv2
from . import label_pb2
import numpy as np
from tabulate import tabulate
from collections import Counter
from pycocotools.coco import COCO

Label = label_pb2.Label()
 
class SegmentationMetric():
    '''
    Args:
        pred_label_path (str): the path to the predict file('*.json') or file dir with file format ('*.txt','*.aqlabel').
        label_path (str): the path to the label file('*.json') or file dir with label file format ('*.txt','*.aqlabel').
        img_path (str): the path to the image dir.
        class_names (list): the list of the class names with training order,without the background. 
    Metrics:
        aAcc: (TP + TN) / (TP + TN + FP + FN), The proportion of the correct number of pixels in the predicted category to the total number of pixels for all images.
        IoU: TP / (TP + FP + FN), The ratio of intersection and union between predicted results and true values.
        Acc: TP / (TP + FN), also named Recall.
        Precision: TP / (TP + FP), The proportion of true predicted values to the total predicted value.
        Dice: Dice coefficient is a set similarity measure function.
    '''
    def __init__(self, pred_label_path='', 
                 label_path='',
                 img_path='', 
                 save_path='',
                 class_names=[], 
                 img_ext='.jpg',
                 fn_threshod=0.5,
                 fp_threshod=0.5,
                 ignore_index=255, 
                 min_size=160,
                 metrics=['aAcc','IoU','Acc','Precision','Dice','CheckBad']):
        super().__init__()
        self.class_names = class_names
        self.num_classes =  len(class_names)
        self.pred_label_path = pred_label_path
        self.label_path = label_path
        self.img_path = img_path
        self.save_path = save_path
        self.fn_threshod = fn_threshod
        self.fp_threshod = fp_threshod
        self.ignore_index = ignore_index
        self.min_size = min_size
        self.metrics = metrics
        self.img_infos = []
        self.results = []
        self.palette = [list(np.random.choice(range(256), size=3)) for _ in range(self.num_classes)]

    def compute_metrics(self):
        self.process()
        results = tuple(zip(*self.results))
        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        headers = []
        datas = []
        ret_metrics = []
        for metric in self.metrics:
            if metric == 'IoU':
                headers.append('mIoU')
                headers.append('IoU')
                iou = total_area_intersect / (total_area_union+1)
                datas.append([iou.mean()]*self.num_classes)
                datas.append(list(iou))
            elif metric == 'aAcc':
                headers.append('aAcc')
                all_acc = total_area_intersect.sum() / total_area_label.sum()
                datas.append([all_acc]*self.num_classes)
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
            elif metric == 'CheckBad':
                fn_path = os.path.join(self.save_path,'fn')
                fp_path = os.path.join(self.save_path,'fp')
                os.makedirs(fn_path,exist_ok=True)
                os.makedirs(fp_path,exist_ok=True)
                for img_info,result in zip(self.img_infos,self.results): 
                    area_intersect, area_union, area_pred_label, area_label = result
                    acc = area_intersect / (area_label+1)
                    acc[area_label==0] = 1
                    pre = area_intersect / (area_pred_label+1)
                    pre[area_pred_label==0] = 1
                    img_path = img_info['filename']
                    image = cv2.imread(img_path)
                    pred_label = img_info['pred_label']
                    label = img_info['label']
                    
                    if min(acc) < self.fn_threshod: # missed detection
                        ids = np.nonzero(acc<self.fn_threshod)[0]
                        for cls_id in ids:
                            if area_label[cls_id] < self.min_size:
                                continue
                            pred = pred_label[label==cls_id+1]
                            pred_ids = list(Counter(pred))
                            
                            pred_id = pred_ids[1] if len(pred_ids)>1 and pred_ids[0]==0 else pred_ids[0]
                            pred_name = self.class_names[pred_id-1] if pred_id!=0 else 'background'
                            class_path = os.path.join(fn_path,self.class_names[cls_id],pred_name)
                            if pred_name==self.class_names[cls_id]:
                                class_path = os.path.join(fn_path,self.class_names[cls_id])
                            pred_img = self.putMask(image,pred_label,self.palette,pred_id) if pred_id != 0 else image.copy()
                            gt_img = self.putMask(image,label,self.palette,cls_id+1)
                            res = np.concatenate([gt_img,pred_img],1)
    
                            os.makedirs(class_path,exist_ok=True)
                            cv2.imwrite(os.path.join(class_path,os.path.basename(img_path)),res)

                    if min(pre) < self.fp_threshod: # over detection
                        cls_id = int(np.argmin(pre))
                        if area_pred_label[cls_id] < self.min_size:
                            continue
                        class_path = os.path.join(fp_path,self.class_names[cls_id])
                        pred_img = self.putMask(image,pred_label,self.palette,cls_id+1)
                        gt_img = self.putMask(image,label,self.palette,cls_id+1)
                        res = np.concatenate([gt_img,pred_img],1)

                        os.makedirs(class_path,exist_ok=True)
                        cv2.imwrite(os.path.join(class_path,os.path.basename(img_path)),res)
                        

        for i,line in enumerate(zip(*datas)):
            ret_metrics.append([self.class_names[i],]+list(line))      
        print(tabulate(ret_metrics,headers))    

    def process(self):
        if not os.path.exists(self.pred_label_path):
            raise Exception(f"The path {self.pred_label_path} of pred does not exist.")
        if not os.path.exists(self.label_path):
            raise Exception(f"The path {self.pred_label_path} of label does not exist.")
        
        pred_gen = self.parse_format(self.pred_label_path)
        gt_gen = self.parse_format(self.label_path)
        for filename in pred_gen:
            pred_label = pred_gen[filename]
            label = gt_gen.get(filename,np.empty(0))
            ext = self.get_ext(self.img_path)
            image_path = os.path.join(self.img_path,filename+ext)
            if not (os.path.exists(image_path) and label.any()):
                continue
            self.img_infos.append({'filename':image_path,'label':label,'pred_label':pred_label})
            self.results.append(self.intersect_and_union(pred_label,label,self.num_classes,self.ignore_index))

    def parse_format(self,data_path):  
        ext = self.get_ext(data_path)
        mask_gen = None

        if ext == '.txt':
            mask_gen = self.yolo_mask(data_path)                  
        elif ext == '.aqlabel':
            mask_gen = self.aqlabel_mask(data_path)
        elif ext == '.png':
            mask_gen = self.gray_mask(data_path)
        elif ext == '.json':
            mask_gen = self.coco_mask(data_path)
        else:
            raise Exception(f"The file format {ext} is not allowed, please supply txt or aqlabel format file.")  
        return mask_gen


    def get_ext(self,path):
        if os.path.isdir(path):
            for f in os.scandir(path):
                return self.get_ext(f.path)
        else:
            return os.path.splitext(path)[1]

    def aqlabel_mask(self,path):
        assert os.path.isdir(path), 'Please give the dir path to aqlabel!'
        def get_mask(file): 
            filename = os.path.splitext(file.name)[0] 
            with open(file.path, "rb") as f:
                label_bin = f.read()
                lb = Label.FromString(label_bin)
                width = int(lb.img_size.width)
                height = int(lb.img_size.height)
                mask = np.zeros((height,width),dtype=np.uint8)
                for r in lb.regions:
                    cls_id = self.class_names.index(r.name)
                    points = []
                    for p in r.polygon.outer.points:
                        points.append([p.x, p.y])
                    polygon = np.array(points,np.int32).reshape(-1,2)
                    cv2.fillConvexPoly(mask, polygon, 1+cls_id)
                return filename,mask
        mask_generator = dict(get_mask(f) for f in os.scandir(path))
        return mask_generator
    
    def gray_mask(self,path):
        assert os.path.isdir(path), 'Please give the dir path to gray mask!'
        def get_mask(file):
            filename = os.path.splitext(file.name)[0] 
            img = cv2.imread(file.path,0)
            return filename,img
        mask_generator = dict(get_mask(f) for f in os.scandir(path))
        return mask_generator

    def yolo_mask(self,path):
        assert os.path.isdir(path), 'Please give the dir path to yolo label!'
        def get_mask(file): 
            filename = os.path.splitext(file.name)[0] 
            ext = self.get_ext(self.img_path)
            image_path = os.path.join(self.img_path,filename+ext)
            img = cv2.imread(image_path)
            height,width = img.shape[:2]
            mask = np.zeros((height,width),dtype=np.uint8)
            with open(file) as f:
                lines = f.readlines()
                for line in lines:
                    item = line.strip().split()
                    cls_id = int(item[0])
                    polygon = np.array(item[1:],dtype=np.float32).reshape(-1,2)
                    polygon[:,0] *= width
                    polygon[:,1] *= height
                    polygon = polygon.astype(np.int32)
                    cv2.fillConvexPoly(mask, polygon, 1+cls_id)
                return filename,mask
        mask_generator = dict(get_mask(f) for f in os.scandir(path))
        return mask_generator

    def coco_mask(self,ann_path):
        coco=COCO(ann_path)
        catIds = coco.getCatIds()
        imgIds = coco.getImgIds()
        def get_mask(imgId,catIds):
            img = coco.loadImgs(imgId)[0]
            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
            filename = os.path.splitext(img['file_name'])[0]
            if len(annIds) > 0:
                mask = coco.annToMask(anns[0]) * (anns[0]['category_id']+1)
                for i in range(len(anns) - 1):
                    mask += coco.annToMask(anns[i + 1]) * (anns[i + 1]['category_id']+1)
                return filename,mask
            else:
                return filename,np.zeros((int(img['height']),int(img['width'])),np.uint8)
        mask_generator = dict(get_mask(imgId,catIds) for imgId in imgIds)
        return mask_generator
            
    @staticmethod
    def putMask(image,mask,palette,class_id,alpha=0.7): 
        result = image.copy()
        class_id = int(class_id)
        if isinstance(class_id,int):
            class_id = [class_id]
        for id_ in class_id:
            result[mask==id_] = result[mask==id_]*(1-alpha)+np.array([[palette[id_-1]]])*alpha
        return result

    @staticmethod
    def intersect_and_union(pred_label, label, num_classes, ignore_index=255):
        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = np.histogram(intersect, bins=num_classes, range=(1,num_classes+1))[0]
        area_pred_label = np.histogram(pred_label, bins=num_classes, range=(1,num_classes+1))[0]
        area_label = np.histogram(label, bins=num_classes, range=(1,num_classes+1))[0]
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label