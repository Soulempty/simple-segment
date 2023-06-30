import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from models.registry import DATASETS

@DATASETS.register_module
class CFDataset(Dataset):
    def __init__(self, dataset_root=None, transforms=None, mode='train', anno_file="train.txt",start_epoch=0):
        self.start_epoch = start_epoch
        self.transforms = transforms
        self.image_list = []
        self.label_list = []
        self.label_ids = []
        self.crop_size = 256
        self.exist_label = True
        self.mode = mode

        anno_path = os.path.join(dataset_root,anno_file)
        with open(anno_path) as fp:
            for line in fp.readlines():
                item = line.strip().split()
                img_path = os.path.join(dataset_root,item[0])
                self.image_list.append(img_path)
                if self.exist_label and len(item)==1:
                    self.exist_label = False
                if self.exist_label:
                    lab_path = os.path.join(dataset_root,item[1])
                    self.label_list.append(lab_path)
        if mode == 'train':
            self.img_mean = np.zeros(3,dtype=np.float32)
            self.img_std = np.zeros(3,dtype=np.float32)
            self.info = self.analyze_data()
            self.cls_weight = np.ones(len(self.info)+1,dtype=np.float32) 
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path = self.image_list[index] 
        result = dict(img_path=img_path)
        result['img'] = cv2.imread(img_path)[:,:,::-1]  # bgr2rgb shallow copy

        if self.exist_label:
            lab_path = self.label_list[index]
            result['label'] = cv2.imread(lab_path,0)  # gray img
        else:
            result['label'] = np.zeros(result['img'].shape[:2],np.uint8)
        if self.mode == 'train':
            weights = self.cls_weight[self.label_ids[index]]
            result['cls_id'] = np.random.choice(self.label_ids[index],1,p=weights/weights.sum())
            result['crop_size'] = self.crop_size
        result['mean'] = self.img_mean
        result['std'] = self.img_std
        if self.transforms:
            result = self.transforms(result)
        return result
    
    def compute_class_weights(self,histogram):

        normHist = histogram / np.sum(histogram)
        for i in range(len(normHist)):
            self.cls_weight[i] = 1 / (np.log(1.1 + normHist[i]))

    def analyze_data(self):
        info = dict()
        rects = []
        print("========start to analyze the train dataset============")
        if self.exist_label:
            for index,(lab_path,img_path) in enumerate(zip(self.label_list,self.image_list)):
                image = cv2.imread(img_path)
                self.img_mean[0] += np.mean(image[:, :, 2])/255.0
                self.img_mean[1] += np.mean(image[:, :, 1])/255.0
                self.img_mean[2] += np.mean(image[:, :, 0])/255.0
                self.img_std[0] += np.std(image[:, :, 2])/255.0
                self.img_std[1] += np.std(image[:, :, 1])/255.0
                self.img_std[2] += np.std(image[:, :, 0])/255.0
                
                label = cv2.imread(lab_path,0)
                ids = list(np.unique(label))
                if len(ids)>1:
                    ids = ids[1:]
                self.label_ids.append(ids)
                for id_ in ids:
                    if id_ == 0:
                        continue
                    mask = np.zeros(label.shape,np.uint8)
                    mask[label==id_] = 1
                    contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    for cont in contours:
                        w, h = cv2.boundingRect(cont)[2:]
                        r = (w*h)**0.5
                        if r>20:
                            rects.append(r)
                    info[id_] = info.get(id_,[])
                    info[id_].append(index)
        self.img_mean /= len(self)
        self.img_std /= len(self)
        self.img_mean = [round(x,6) for x in self.img_mean.tolist()]
        self.img_std =  [round(x,6) for x in self.img_std.tolist()]
        print("mean:",self.img_mean, " std:",self.img_std)

        rects = np.array(rects)
        y = (rects/128+0.5).astype(np.int32)
        m = y.max()
        hist = np.histogram(y,bins=m+1)[0]
        self.crop_size = 128*max((hist.argmax()+m+1)//2,2)
        print("crop_size:", self.crop_size)
        return info 
    
    def get_weight(self,score):
        weights = [0]*len(self)
        w = np.array(score)
        w = 1/(w+0.1)  # weight for each class.
        self.cls_weight = w
        n = (score<score.mean()).sum()
        ids = np.argsort(w)[-n:]
        for id_ in ids:
            if id_ !=0:
                indexs = self.info[id_]
                for index in indexs:
                    weights[index] = w[id_]
        return weights