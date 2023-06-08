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
            config_path = 'infer.yaml'

        self.config = Config(config_path)
        self.model_path = self.config.model_path
        self.input_size = self.config.input_size
        self.palette = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for i in range(len(self.config.class_names))]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.run_mode = 'torch'

    def predict(self,inputs=[]):
        results = {}
        index = 0
        duration = 0
        for item in tqdm(inputs):
            index += 1
            image = item
            basename = str(index)+'.png'
            mask = np.zeros(self.input_size,np.uint8)
            if isinstance(item,str) and os.path.exists(item):
                basename = os.path.basename(item).split('.')[0]+'.png'
                image = cv2.imread(item)

            input = self.img_proc(image)
            torch.cuda.synchronize()
            st = time.time()
            if self.run_mode == 'trt':
                outputs = self.model.infer(input)
                mask = outputs[0].reshape(*self.input_size).astype(np.uint8)  
            else:
                with torch.no_grad():
                    input = torch.from_numpy(input).unsqueeze(0)
                    input = input.to(self.device)
                    output = self.model(input)[0]
                    mask = torch.argmax(output,dim=1)[0].cpu().numpy().astype(np.uint8)
            torch.cuda.synchronize()
            ed = time.time()
            duration += ed - st
            result = self.pre_to_img(image,mask)
            results[basename] = result
        print(f"Infer time for each image:{duration/len(inputs)} s")
        return results
            
    def load_model(self,model_path=None):
        model_path = model_path if model_path else self.model_path
        if model_path.endswith('.pth') or model_path.endswith('.pt'):
            self.model = build_module(self.config.model)
            weight = torch.load(model_path, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(weight)
            self.model.to(self.device)
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
    
    def set_input_size(self,input_size=None):
        if input_size:
            self.input_size = input_size

    def img_proc(self,image):
        img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(self.input_size[1],self.input_size[0]),interpolation=cv2.INTER_LINEAR)
        input = img.astype(np.float32)/255.0
        input = input.transpose(2, 0, 1)
        return input
    
    def pre_to_img(self,img,mask,alpha=0.5):
        h,w = img.shape[:2]
        mask = cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)
        ids = np.unique(mask)
        for id_ in ids:
            if id_ == 0:
                continue
            img[mask==id_] = np.array([self.palette[id_]])*alpha + img[mask==id_]*(1-alpha) 
        return img
        
