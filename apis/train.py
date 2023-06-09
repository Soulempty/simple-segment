import os
import copy
import time
import torch
import yaml
import codecs
from models import Segment_
import numpy as np

import torch.optim as optim
from torch.utils.data import DataLoader

from .config import Config
from .utils import evaluate,async_,set_seed
from .build import build_module
#from models import Segment_

class Segment():
    def __init__(self,cfg_path=None,seed=256):
        super(Segment, self).__init__()
        self.device = self.set_device()
        set_seed(seed)
        self.cfg_path = "config.yaml"
        if cfg_path:
            self.cfg_path = cfg_path       
        self.config = Config(self.cfg_path)
        self._batch_size = self.config.batch_size
        self._epochs = self.config.epochs
        self._work_dir = self.config.work_dir
        self.log_iters = self.config.log_iters
        self._model_dir = os.path.join(self.work_dir,'model')
        self._best_model_path = ''
        os.makedirs(self.model_dir,exist_ok=True)
        self._class_names = self.config.class_names
        self._num_workers = self.config.num_workers
        self._lr = self.config.optimizer['lr']
        self._base_size = self.config.train_dataset['transforms']['augmentations'][0]['base_size']
        self._iters_per_epoch = 0
        self._iters = 0
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.criterion = None

        self.best_mean_iou = -1.0
        self._best_model_iter = -1
        self._iter = 0
        self._epoch = 0
        self.step_loss = 0
        self._stop_flag = False
        self.flag = 0
        self.init_config()

    def init_config(self):
        
        self.train_dataset = build_module(self.config.train_dataset)
        self.val_dataset = build_module(self.config.val_dataset)
        self.train_loader = DataLoader(self.train_dataset,self.batch_size,shuffle=True,num_workers=self.num_workers)
        self._iters_per_epoch = len(self.train_dataset)//self.batch_size
        self._iters = self._epochs*self.iters_per_epoch
        self.log_iters = min(self._iters_per_epoch,self.log_iters)
        
        self.model = build_module(self.config.model).to(self.device)
        self.criterion = build_module(self.config.loss)

        optimizer_cfg = copy.deepcopy(self.config.optimizer)
        self.optimizer = getattr(optim,optimizer_cfg.pop('type'))(self.model.parameters(),**optimizer_cfg) 
 
        lr_scheduler_cfg = copy.deepcopy(self.config.lr_scheduler)
        if isinstance(lr_scheduler_cfg,list):
            schedulers = []
            for cfg in lr_scheduler_cfg:
                scheduler_type = cfg.pop('type')
                if scheduler_type == "PolynomialLR":
                    cfg['total_iters'] = self.iters
                if scheduler_type == "LinearLR":
                    cfg['total_iters'] = min(cfg['total_iters'],self.iters_per_epoch*5)
                schedulers.append(getattr(optim.lr_scheduler,scheduler_type)(self.optimizer,**cfg))
            self.lr_scheduler = optim.lr_scheduler.ChainedScheduler(schedulers)
        else:
            scheduler_type = lr_scheduler_cfg.pop('type')
            if scheduler_type == "PolynomialLR":
                lr_scheduler_cfg['total_iters'] = self.iters
            self.lr_scheduler = getattr(optim.lr_scheduler,scheduler_type)(self.optimizer,**lr_scheduler_cfg)

        cfg_path = os.path.join(self.work_dir,'config.yaml')
        with codecs.open(cfg_path, 'w', 'utf-8') as file:
            yaml.dump(self.config.data, file, sort_keys=False)

    @async_
    def train(self,call_back=None):
        
        print("-------- TRAINING  BEGINING ------------")
        avg_loss = 0
        while self.iter < self.iters:
            st = time.time()
            for data in self.train_loader:
                self._iter += 1
                if (self.iter > self.iters) or self.stop_flag:
                    break
                images = data['img'].to(self.device)
                labels = data['label'].to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(images)
                loss = sum([self.criterion(logit,labels) for logit in logits])
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                avg_loss += loss.item()
                self.step_loss += loss.item()
                if self.iter%self.log_iters == 0:
                    avg_loss /= self.log_iters
                    lr = self.lr_scheduler.get_last_lr()[0]
                    print("[TRAIN] epoch: {}, iter: {}/{}, loss: {:.4f}, lr: {:.6f}".format((self.iter - 1) // self.iters_per_epoch + 1, self.iter, self.iters, avg_loss,lr))
                    avg_loss = 0
            if self.stop_flag:
                self.flag = -1
                print("finishing stopping training...")
                break
            self._epoch += 1
            self.step_loss /= self.iters_per_epoch
            lr = self.lr_scheduler.get_last_lr()[0]
            if self.val_dataset is not None:
                test_mean_iou, aAcc = evaluate(self.model,self.val_dataset,self.class_names,self.device)
                if test_mean_iou > self.best_mean_iou:
                    self.best_mean_iou = test_mean_iou.item()
                    self._best_model_iter = self.iter
                    self._best_model_path = os.path.join(self.model_dir, f'best.pth')
                    torch.save(self.model.state_dict(), self.best_model_path)
                duration = time.time()-st
                left_time = duration*(self.epochs-self.epoch)
                content = {'iter':self.iter,'epoch':self.epoch,'left_time':left_time,'loss':self.step_loss,'lr':lr,'test_mean_iou':test_mean_iou.item()}
                if call_back:
                    call_back(content)
            self.step_loss = 0.0
        self.export()
        print("-------- TRAINING  ENDINING ------------")
    
    def export(self):
        onnx_path = os.path.join(self.model_dir,'model.onnx')
        print(f"exporting the onnx model to path {onnx_path}")
        input = torch.randn(size=(1,3,*self.base_size)).to(self.device)
        model = Segment_(self.model)
        model.eval()
        input_names = ["input"] 
        output_names = ["output"]
        torch.onnx.export(model, input,onnx_path, 
                          opset_version=12,
                          do_constant_folding=True,
                          input_names=input_names, 
                          output_names=output_names,
                          verbose=False)
        print("-------- finishing export onnx model ------------")

    def set_device(self):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def set_lr(self,lr):
        self.config.optimizer['lr'] = lr
        self.init_config()

    def set_epoch(self,epoch):
        self._epochs = epoch
        self.config.epochs = epoch
        self.init_config()
        
    def set_datapath(self,data_path):
        self.config.train_dataset['dataset_root'] = data_path
        self.config.val_dataset['dataset_root'] = data_path
        self.init_config()

    def set_basesize(self,base_size):
        self._base_size = base_size
        self.config.train_dataset['transforms']['augmentations'][0]['base_size'] = base_size
        self.init_config()
    
    def set_cropsize(self,crop_size):
        self.config.train_dataset['transforms']['augmentations'][1]['crop_size'] = crop_size
        self.init_config()
    
    def set_classnames(self,class_names):
        self._class_names = class_names
        self.init_config()
    
    def set_workdir(self,work_dir):
        self._work_dir = work_dir
        self.init_config()

    @property
    def batch_size(self):
        return self._batch_size
    @property
    def base_size(self):
        return self._base_size
    @property
    def model_dir(self):
        return self._model_dir
    @property
    def best_model_path(self):
        return self._best_model_path
    @property
    def work_dir(self):
        return self._work_dir
    @property
    def epochs(self):
        return self._epochs
    @property
    def epoch(self):
        return self._epoch
    @property
    def iter(self):
        return self._iter
    @property
    def best_model_iter(self):
        return self._best_model_iter
    @property
    def lr(self):
        return self._lr
    @property
    def iters(self):
        return self._iters
    @property
    def iters_per_epoch(self):
        return self._iters_per_epoch
    @property
    def class_names(self):
        return self._class_names
    @property
    def num_workers(self):
        return self._num_workers
    @property
    def stop_flag(self):
        return self._stop_flag

    def set_flag(self):
        self._stop_flag = True
        return 0
    
    def stop(self):
        self.set_flag()
        while True:
            if self.flag == -1:
                print("flag:",self.flag)
                break

    
