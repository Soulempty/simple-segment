import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from .registry import DATASETS

@DATASETS.register_module
class CFDataset(Dataset):
    def __init__(self, dataset_root=None, transforms=None, anno_file="train.txt"):
        self.transforms = transforms
        self.image_list = []
        self.label_list = []
        self.exist_label = True

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
        if self.transforms:
            result = self.transforms(result)
        return result