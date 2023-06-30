import os
import copy
import time
import torch
import yaml
import codecs
from models import Segment_
import numpy as np

import torch.optim as optim
from torch.utils.data import DataLoader,WeightedRandomSampler

from .config import Config
from .utils import *
from .build import build_module

class Segment():
    def __init__(self,cfg_path=None,seed=256):
        super(Segment, self).__init__()
        self.device = self.set_device()
        set_seed(seed)
        self.cfg_path = "configs/config.yaml"
        if cfg_path:
            self.cfg_path = cfg_path       
        self.config = Config(self.cfg_path)
        self.resume = self.config.resume
        self.use_finetune = getattr(self.config,'use_finetune',False)
        self.use_weightsampler = getattr(self.config,'use_weightsampler',False)
        self._batch_size = self.config.batch_size
        self._epochs = self.config.epochs
        self._work_dir = self.config.work_dir
        self.log_iters = self.config.log_iters
        self._model_dir = os.path.join(self.work_dir,'model')
        self._best_model_path = os.path.join(self.model_dir, f'best.pth')
        self.config.data['model_path'] = self.best_model_path
        self._class_names = self.config.class_names
        self._num_classes = len(self._class_names)
        self._num_workers = self.config.num_workers
        self._lr = self.config.optimizer['lr']
        self._base_size = self.config.train_dataset[0]['transforms']['augmentations'][0]['base_size']
        self.dataset_epochs = [data['start_epoch'] for data in self.config.train_dataset]
        self.config.data['input_size'] = self.base_size
        self.rebalance_epoch = self.config.rebalance_epoch
        self._iters_per_epoch = 0
        self._iters = 0
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.criterion = None

        self.best_mean_iou = -1.0
        self._best_model_epoch = -1
        self._iter = 0
        self._epoch = 0
        self.start_epoch = 0
        self.step_loss = 0
        self._stop_flag = False
        self.flag = 0

    def init_config(self):
        os.makedirs(self.model_dir,exist_ok=True)
        self.init_dataset()
        
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

        if self.resume:
            if os.path.isfile(self.resume):
                checkpoint = torch.load(self.resume)
                self._epoch = checkpoint['epoch']
                self._iter = self.epoch * self.iters_per_epoch
                self._best_model_epoch = self.epoch
                self.model.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                print("=====> loaded checkpoint '{}' (epoch {})".format(self.resume, checkpoint['epoch']))
            else:
                print("=====> no checkpoint found at '{}'".format(self.resume))

        cfg_path = os.path.join(self.work_dir,'config.yaml')
        
        with codecs.open(cfg_path, 'w', 'utf-8') as file:
            yaml.dump(self.config.data, file, sort_keys=False)

    def init_dataset(self,index=0):
        print(f"init the {index} dataset {self.config.train_dataset[index]['type']}")
        self.train_dataset = build_module(self.config.train_dataset[index])
        self.val_dataset = build_module(self.config.val_dataset)
        self.train_loader = DataLoader(self.train_dataset,self.batch_size,shuffle=True,num_workers=self.num_workers,drop_last=True)
        self._iters_per_epoch = len(self.train_dataset)//self.batch_size
        self._iters = self._epochs*self.iters_per_epoch
        self.log_iters = min(self._iters_per_epoch,self.log_iters)
        img_mean = self.train_dataset.img_mean
        img_std = self.train_dataset.img_std
        self.config.train_dataset[index]['transforms']['augmentations'][-1]['mean'] = img_mean
        self.config.train_dataset[index]['transforms']['augmentations'][-1]['std'] = img_std
        self.val_dataset.img_mean = img_mean
        self.val_dataset.img_std = img_std
        self.config.data['mean'] = img_mean
        self.config.data['std'] = img_std
        if index >0:
            cfg_path = os.path.join(self.work_dir,'config.yaml')
            with codecs.open(cfg_path, 'w', 'utf-8') as file:
                yaml.dump(self.config.data, file, sort_keys=False)

    @async_
    def train(self,call_back=None):
        
        print("-------- TRAINING  BEGINING ------------")
        self.model.train()
        if self.resume:
            print("==========> evaluate the resume model.")
            test_mean_iou, cls_iou = evaluate(self.model,self.val_dataset,self.class_names,self.device)
            self.best_mean_iou = test_mean_iou.item()
        avg_loss = 0
        while self.iter < self.iters:
            st = time.time()
            self.set_cropsize() # random crop size.
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
            if self.epoch in self.dataset_epochs:
                index = self.dataset_epochs.index(self.epoch)
                self.init_dataset(index)
            self.step_loss /= self.iters_per_epoch
            lr = self.lr_scheduler.get_last_lr()[0]
            if self.val_dataset is not None :
                test_mean_iou, cls_iou = eval(self.config.evaluate)(self.model,self.val_dataset,self.class_names,self.device)
                if test_mean_iou > self.best_mean_iou:
                    self.best_mean_iou = test_mean_iou.item()
                    self._best_model_epoch = self.epoch
                    state = {"epoch": self.epoch, "model": self.model.state_dict(),'optimizer':self.optimizer.state_dict(),'lr_scheduler':self.lr_scheduler.state_dict()}
                    torch.save(state, self.best_model_path)
                duration = time.time()-st
                remain_time = duration*(self.epochs-self.epoch)
                m, s = divmod(remain_time, 60)
                h, m = divmod(m, 60)
                print("Remaining training time = %d hour %d minutes %d seconds" % (h, m, s))
                content = {'iter':self.iter,'epoch':self.epoch,'remain_time':remain_time,'loss':self.step_loss,'lr':lr,'test_mean_iou':test_mean_iou.item()}
                if call_back:
                    call_back(content)
                if self.epochs-self.epoch <= self.rebalance_epoch:
                    print("---------begining the reblance for classes by weighted sampler----------")
                    if self.use_finetune:
                        weights = self.train_dataset.get_weight(cls_iou)
                        self.criterion.class_weight = self.train_dataset.cls_weight
                        if self.use_weightsampler:
                            weighted_sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
                            self.train_loader = DataLoader(self.train_dataset,self.batch_size,shuffle=False,sampler=weighted_sampler,num_workers=self.num_workers)
            self.step_loss = 0.0
        print(f"=========The best mIoU is {self.best_mean_iou} at epoch {self.best_model_epoch}.=============")
        self.export_engine()
        print("-------- TRAINING  ENDINING ------------")
    
    def export_onnx(self):
        onnx_path = os.path.join(self.model_dir,'model.onnx')
        print(f"exporting the onnx model to path {onnx_path}")
        input = torch.randn(size=(1,3,*(self.base_size[::-1]))).to(self.device)
        weight = torch.load(self.best_model_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(weight['model'])
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
        return onnx_path
    
    def export_engine(self,workspace=4,half=True):
        onnx_path = self.export_onnx()
        self.release()
        import tensorrt as trt
        engine_path = os.path.join(self.model_dir,'model.engine')
        print(f"exporting the trt model to path {engine_path }")
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.max_workspace_size = workspace * 1 << 30
        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        parser.parse_from_file(onnx_path)
        if builder.platform_has_fast_fp16 and half:
            config.set_flag(trt.BuilderFlag.FP16)
        with builder.build_engine(network, config) as engine, open(engine_path, 'wb') as f:
            f.write(engine.serialize())  
        print("-------- finishing export trt model ------------")       

    def set_device(self):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_lr(self,lr=None):
        if lr:
            self.config.optimizer['lr'] = lr

    def set_epoch(self,epoch=None):
        if epoch:
            self._epochs = epoch
            self.config.epochs = epoch
        
    def set_datapath(self,data_path=None):
        if data_path:
            self.config.train_dataset['dataset_root'] = data_path
            self.config.val_dataset['dataset_root'] = data_path

    def set_basesize(self,base_size=None):
        if base_size:
            self._base_size = base_size
            self.config.train_dataset['transforms']['augmentations'][0]['base_size'] = base_size
    
    def set_classnames(self,class_names=None):
        if class_names:
            self._class_names = class_names
            self._num_classes = len(class_names)
            self.config.model['num_classes'] = self.num_classes
    
    def set_workdir(self,work_dir=None):
        if work_dir:
            self._work_dir = work_dir
            self._model_dir = os.path.join(self.work_dir,'model')
            self._best_model_path = os.path.join(self.model_dir, f'best.pth')
            self.config.model_path = self.best_model_path.replace('.pth','.engine')
            self.config.work_dir = work_dir
    
    def set_batchsize(self,batch_size=None):
        if batch_size:
            self._batch_size = batch_size
            self.config.batch_size = batch_size

    def set_cropsize(self):
        start = 256
        stop = max(min(self.base_size)//128*128,256)
        crop_size = np.linspace(start,stop,(stop-start)//128+1,dtype=int)
        self.train_dataset.crop_size = np.random.choice(crop_size)
        return crop_size

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
    def num_classes(self):
        return self._num_classess
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
    def best_model_epoch(self):
        return self._best_model_epoch
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
    
    def release(self):
        try:
            del self.train_dataset
            del self.val_dataset 
            del self.train_loader
            self.model = self.model.eval().cpu()
            del self.optimizer
            del self.lr_scheduler
            del self.model
        except Exception as e:
            print(f"{e} occured while the optimizer and model are released!")
        torch.cuda.empty_cache() 
        torch.cuda.empty_cache() 
        time.sleep(2)
        print("===========finishing releasing the memory=================")
        return
    def stop(self):
        self.set_flag()
        while True:
            if self.flag == -1:
                print("flag:",self.flag)
                break

    
