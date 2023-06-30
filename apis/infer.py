import os
import cv2
import time
import torch
from tqdm import tqdm
import numpy as np
from .trt import TRT
from .build import build_module
from .config import Config

class SegmentModel():
    def __init__(self, config_path=None):
        super().__init__()
        if config_path == None:
            config_path = 'configs/infer.yaml'

        self.config = Config(config_path)
        self.model_path = self.config.model_path if os.path.exists(self.config.model_path) else os.path.join(os.path.dirname(config_path),'model/best.pth')
        self.img_mean = np.array(self.config.mean,np.float32)
        self.img_std = np.array(self.config.std,np.float32)
        self.input_size = self.config.input_size[::-1] # h,w
        self.class_names = self.config.class_names
        self.num_classes = len(self.class_names)
        self.get_palette()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.run_mode = 'torch' 

    def predict(self,input=None):
        image = input
        mask = np.zeros(self.input_size,np.uint8)
        if isinstance(input,str) and os.path.exists(input):
            image = cv2.imread(input)
        input = self.img_proc(image)
        if self.run_mode == 'trt':
            outputs = self.model.infer(input)
            mask = outputs[0].reshape(*self.input_size).astype(np.uint8)  
        else:
            with torch.no_grad():
                input = torch.from_numpy(input).unsqueeze(0)
                input = input.to(self.device)
                output = self.model(input)[0]
                mask = torch.argmax(output,dim=1)[0].cpu().numpy().astype(np.uint8)
        h,w = image.shape[:2]
        result = cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)
        return result
            
    def load_model(self,model_path=None):
        model_path = model_path if model_path else self.model_path
        if model_path.endswith('.pth') or model_path.endswith('.pt'):
            self.model = build_module(self.config.model)
            self.model.to(self.device)
            weight = torch.load(model_path, map_location=self.device)
            if 'optimizer' in weight:
                self.model.load_state_dict(weight['model'])
            else:
                self.model.load_state_dict(weight)
            self.model.eval()
            with torch.no_grad():
                input = torch.randn([1,3,*self.input_size]).to(self.device)
                output = self.model(input)[0]
        elif model_path.endswith('.trt') or model_path.endswith('.engine'):
            self.model = TRT(model_path,self.input_size) # h,w
            input = np.random.random((3,*self.input_size)).astype(np.float32)
            output = self.model.infer(input)[0]
            self.run_mode = 'trt'

    def set_model_path(self,model_path=None):
        if model_path:
            self.model_path = model_path
    
    def set_num_classes(self,num_classes):
        self.num_classes = num_classes
        self.config.model['num_classes'] = num_classes
        self.get_palette()

    def set_input_size(self,input_size=None):
        if input_size:
            self.input_size = input_size

    def img_proc(self,image):
        img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(self.input_size[1],self.input_size[0]),interpolation=cv2.INTER_LINEAR)
        input = (img.astype(np.float32)/255.0-self.img_mean)/self.img_std
        input = input.transpose(2, 0, 1)
        return input
    
    def get_palette(self):
        self.palette = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for i in range(self.num_classes)]

    def pre_to_img(self,img,mask,alpha=0.7):
        assert img.shape[:2] == mask.shape[:2]
        ids = np.unique(mask)
        for id_ in ids:
            if id_ == 0:
                continue
            img[mask==id_] = np.array([self.palette[id_]])*alpha + img[mask==id_]*(1-alpha) 
        return img
        
