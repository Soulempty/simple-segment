
import cv2
import numbers
import random
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageEnhance
from models.registry import TRANSFORMS

@TRANSFORMS.register_module
class Resize(object):
    def __init__(self, base_size,keep_ratio=False):
        self.base_size = base_size
        self.keep_ratio = keep_ratio
    def __call__(self, result):
        img = result['img']
        label = result['label']

        new_w,new_h = self.base_size
        h,w = img.shape[:2]
        if self.keep_ratio:
            scale_factor = min(max(new_w,new_h) / max(h, w),min(new_w,new_h) / min(h, w))
            new_w = int(w * scale_factor + 0.5)
            new_h = int(h * scale_factor + 0.5)
        result['img'] = cv2.resize(img,(new_w,new_h), interpolation=cv2.INTER_LINEAR)
        result['label'] = cv2.resize(label,(new_w,new_h), interpolation=cv2.INTER_NEAREST) 

        return result


@TRANSFORMS.register_module
class CopyPaste(object):
    def __init__(self,prob=0.7):
        self.prob = prob
    def __call__(self, result):
        if np.random.random()>self.prob:
            return result

        img = result['img']
        label = result['label']
        cls_id = result['cls_id']
        height,width = img.shape[:2]
        mask = np.zeros((height,width),np.uint8)
        mask[label==cls_id] = 1
        contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for cont in contours:
            x, y, w, h = cv2.boundingRect(cont)
            if (w*h)**0.5<16:
                continue
            mask_block = mask[y:y+h,x:x+w]
            label_block = label[y:y+h,x:x+w]*mask_block
            img_block = img[y:y+h,x:x+w]*mask_block[:,:,np.newaxis]
            n = max(min(height//h,width//w)//4,4)
            while n>0:
                label_b = label_block
                img_b = img_block
                mask_b = mask_block
                if np.random.random()>0.5:
                    label_b = cv2.flip(label_block,1)
                    img_b = cv2.flip(img_block,1)
                    mask_b = cv2.flip(mask_block,1)
                if np.random.random()>0.5:
                    scale_factors = np.linspace(0.75,1.5,26).tolist()
                    if int(w*1.5+0.5)+4>width or int(h*1.5+0.5)+4>height:
                        scale_factors = np.linspace(0.75,1.0,6).tolist()
                    np.random.shuffle(scale_factors)
                    scale_factor = scale_factors[0]
                    label_b = cv2.resize(label_block,None,fx=scale_factor,fy=scale_factor,interpolation=cv2.INTER_NEAREST)
                    mask_b = cv2.resize(mask_block,None,fx=scale_factor,fy=scale_factor,interpolation=cv2.INTER_NEAREST)
                    img_b = cv2.resize(img_block,None,fx=scale_factor,fy=scale_factor,interpolation=cv2.INTER_LINEAR)
                bh,bw = img_b.shape[:2]
                x1 = 0
                y1 = 0
                if width-bw-4>0:
                    x1 = np.random.randint(0,width-bw-4)
                if height-bh-4>0:
                    y1 = np.random.randint(0,height-bh-4)
                if (label[y1:y1+bh,x1:x1+bw]>0).sum()>int(bh*bw*0.1):
                    continue
                n -= 1
                img[y1:y1+bh,x1:x1+bw] = img[y1:y1+bh,x1:x1+bw]*(1-mask_b[:,:,np.newaxis]) + img_b*mask_b[:,:,np.newaxis]
                label[y1:y1+bh,x1:x1+bw] = label_b
        result['img'] = img
        result['label'] = label
        return result   
    
@TRANSFORMS.register_module
class ResizeStepScaling(object):
    """
    Scale an image proportionally within a range.

    Args:
        min_scale_factor (float, optional): The minimum scale. Default: 0.75.
        max_scale_factor (float, optional): The maximum scale. Default: 1.25.
        scale_step_size (float, optional): The scale interval. Default: 0.25.

    Raises:
        ValueError: When min_scale_factor is smaller than max_scale_factor.
    """

    def __init__(self,
                 min_scale_factor=0.75,
                 max_scale_factor=1.25,
                 scale_step_size=0.25):
        if min_scale_factor > max_scale_factor:
            raise ValueError(
                'min_scale_factor must be less than max_scale_factor, '
                'but they are {} and {}.'.format(min_scale_factor,
                                                 max_scale_factor))
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.scale_step_size = scale_step_size

    def __call__(self, result):
        img = result['img']
        label = result['label']
        h,w = img.shape[:2]

        if self.min_scale_factor == self.max_scale_factor:
            scale_factor = self.min_scale_factor

        elif self.scale_step_size == 0:
            scale_factor = np.random.uniform(self.min_scale_factor,
                                             self.max_scale_factor)

        else:
            num_steps = int((self.max_scale_factor - self.min_scale_factor) /
                            self.scale_step_size + 1)
            scale_factors = np.linspace(self.min_scale_factor,
                                        self.max_scale_factor,
                                        num_steps).tolist()
            np.random.shuffle(scale_factors)
            scale_factor = scale_factors[0]
        w = int(round(scale_factor * w))
        h = int(round(scale_factor * h))

        result['img'] = cv2.resize(img,(w, h), interpolation=cv2.INTER_LINEAR)
        result['label'] = cv2.resize(label,(w, h), interpolation=cv2.INTER_NEAREST) 
        return result

@TRANSFORMS.register_module    
class RandomCrop(object):
    def __init__(self, crop_size):
        self.crop_size = (crop_size,crop_size) if isinstance(crop_size, numbers.Number) else crop_size

    def __call__(self, result):
        img = result['img']
        label = result['label']
        assert img.shape[:2] == label.shape[:2]
        tw, th = self.crop_size
        h,w = img.shape[:2]

        if w == tw and h == th:
            return result
        if w < tw or h < th:
            result['img'] = cv2.resize(img,(tw, th), interpolation=cv2.INTER_LINEAR)
            result['label'] = cv2.resize(label,(tw, th), interpolation=cv2.INTER_NEAREST) 

            return result

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        result['img'] = img[y1:y1+th,x1:x1+tw]
        result['label'] = label[y1:y1+th,x1:x1+tw]
        return result

@TRANSFORMS.register_module   
class RandomPaddingCrop:
    def __init__(self,
                 crop_size=(512, 512),
                 im_padding_value=128,
                 label_padding_value=255,
                 ignore_index=255):
        self.crop_size = crop_size
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value
        self.ignore_index = ignore_index

    def __call__(self, result):
        img = result['img']
        label = result['label']
        assert img.shape[:2] == label.shape[:2]
        h,w = img.shape[:2]
        pad_height = max(self.crop_size[1]-h, 0)
        pad_width = max(self.crop_size[0]-w, 0) 
        if (pad_height > 0 or pad_width > 0):
            result['img'] = cv2.copyMakeBorder(img, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=self.im_padding_value) #top,bottom,left,right
            result['label'] = cv2.copyMakeBorder(label, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=self.label_padding_value)
            
        h,w = result['img'].shape[:2]
        if w == self.crop_size[0] and h == self.crop_size[1]:
            return result
        
        x1 = random.randint(0, w - self.crop_size[0])
        y1 = random.randint(0, h - self.crop_size[1])
        result['img'] = img[y1:y1+self.crop_size[1],x1:x1+self.crop_size[0]]
        result['label'] = label[y1:y1+self.crop_size[1],x1:x1+self.crop_size[0]]
        return result

@TRANSFORMS.register_module
class RandomDistort:
    """
    Distort an image with random configurations.

    Args:
        brightness_range (float, optional): A range of brightness. Default: 0.5.
        brightness_prob (float, optional): A probability of adjusting brightness. Default: 0.5.
        contrast_range (float, optional): A range of contrast. Default: 0.5.
        contrast_prob (float, optional): A probability of adjusting contrast. Default: 0.5.
        saturation_range (float, optional): A range of saturation. Default: 0.5.
        saturation_prob (float, optional): A probability of adjusting saturation. Default: 0.5.
        hue_range (int, optional): A range of hue. Default: 18.
        hue_prob (float, optional): A probability of adjusting hue. Default: 0.5.
        sharpness_range (float, optional): A range of sharpness. Default: 0.5.
        sharpness_prob (float, optional): A probability of adjusting saturation. Default: 0.
    """

    def __init__(self,
                 brightness_range=0.5,
                 brightness_prob=0.5,
                 contrast_range=0.5,
                 contrast_prob=0.5,
                 saturation_range=0.5,
                 saturation_prob=0.5,
                 hue_range=18,
                 hue_prob=0.5,
                 sharpness_range=0.5,
                 sharpness_prob=0):
        self.brightness_range = brightness_range
        self.brightness_prob = brightness_prob
        self.contrast_range = contrast_range
        self.contrast_prob = contrast_prob
        self.saturation_range = saturation_range
        self.saturation_prob = saturation_prob
        self.hue_range = hue_range
        self.hue_prob = hue_prob
        self.sharpness_range = sharpness_range
        self.sharpness_prob = sharpness_prob

    def _brightness(self,im,brightness_range):
        brightness_delta = np.random.uniform(1-brightness_range, 1+brightness_range)
        im = ImageEnhance.Brightness(im).enhance(brightness_delta)
        return im
    def _contrast(self,im, contrast_range):
        contrast_delta = np.random.uniform(1-contrast_range,1+contrast_range)
        im = ImageEnhance.Contrast(im).enhance(contrast_delta)
        return im
    def _saturation(self,im, saturation_range):
        saturation_delta = np.random.uniform(1-saturation_range,1+saturation_range)
        im = ImageEnhance.Color(im).enhance(saturation_delta)
        return im
    def _sharpness(self,im, sharpness_range):
        sharpness_delta = np.random.uniform(1-sharpness_range,1+sharpness_range)
        im = ImageEnhance.Sharpness(im).enhance(sharpness_delta)
        return im
    def _hue(self,im, hue_range):
        hue_delta = np.random.uniform(-hue_range,hue_range)
        im = np.array(im.convert('HSV'))
        im[:, :, 0] = im[:, :, 0] + hue_delta
        im = Image.fromarray(im, mode='HSV').convert('RGB')
        return im
    def __call__(self,result):
        img = Image.fromarray(result['img'])
        ops = [self._brightness,self._contrast,self._saturation,self._sharpness,self._hue]
        random.shuffle(ops)
        prob_dict = {
            'brightness': self.brightness_prob,
            'contrast': self.contrast_prob,
            'saturation': self.saturation_prob,
            'hue': self.hue_prob,
            'sharpness': self.sharpness_prob
        }
        param_dict = {
            'brightness': self.brightness_range,
            'contrast': self.contrast_range,
            'saturation': self.saturation_range,
            'hue': self.hue_range,
            'sharpness': self.sharpness_range
        }
        for i in range(len(ops)):
            prob = prob_dict[ops[i].__name__[1:]]
            if np.random.uniform(0, 1) < prob:
                param = param_dict[ops[i].__name__[1:]]
                img = ops[i](img,param)
        result['img'] = np.asarray(img)
        return result        

@TRANSFORMS.register_module
class RandomHorizontalFlip(object):   
    def __call__(self, result):
        if random.random() < 0.5:
            result['img'] = cv2.flip(result['img'],1)
            result['label'] = cv2.flip(result['label'],1)
            return result
        return result 
    

@TRANSFORMS.register_module
class RandomVerticalFlip(object):   
    def __call__(self, result):
        if random.random() < 0.5:
            result['img'] = cv2.flip(result['img'],0)
            result['label'] = cv2.flip(result['label'],0)
            return result
        return result 
    

@TRANSFORMS.register_module
class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0.0, 0.0, 0.0), std=(1.0,1.0,1.0)):
        self.mean = mean
        self.std = std

    def __call__(self, result):
        result['label'] = result['label'].long()
        self.mean = result['mean']
        self.std = result['std']
        result['img'] = F.normalize(result['img'], self.mean, self.std).float()

        return result

@TRANSFORMS.register_module
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, result):
        result['img'] = torch.from_numpy(np.array(result['img'],np.float32).transpose((2, 0, 1))).div(255)
        result['label'] = torch.from_numpy(np.array(result['label'],np.int64))

        return result

@TRANSFORMS.register_module    
class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, result):
        for a in self.augmentations:
            result = a(result)

        return result

@TRANSFORMS.register_module   
class ForegroundCrop(object):
    def __init__(self,crop_size=(256,256),prob=0.9,ignore_index=255):
        super().__init__()
        self.crop_size = crop_size
        self.prob = prob
        self.ignore_index = ignore_index
        self.shift = crop_size[0]//20

    def crop_bbox(self, mask, h, w):
        def generate_global_crop(h,w):
            margin_h = max(h - self.crop_size[1], 0)
            margin_w = max(w - self.crop_size[0], 0)
            ymin = np.random.randint(0, margin_h + 1)
            xmin = np.random.randint(0, margin_w + 1)
            ymax = ymin + self.crop_size[1]
            xmax = xmin + self.crop_size[0]

            return xmin,ymin,xmax,ymax
        
        def generate_local_crop(mask,h,w):
            cls_id = list(np.unique(mask))[1:]
            if len(cls_id)==0:
                generate_global_crop(h,w)
            random_id = random.choice(cls_id)
            loc = np.where(mask==random_id)
            random_index = random.choice(range(loc[0].shape[0]))
            x = loc[1][random_index] + random.randint(-self.shift,self.shift)
            y = loc[0][random_index] + random.randint(-self.shift,self.shift)
            xmin = min(w-self.crop_size[0],x-self.crop_size[0]//2) if x > w//2 else max(0,x-self.crop_size[0]//2)
            ymin = min(h-self.crop_size[1],y-self.crop_size[1]//2) if y > h//2 else max(0,y-self.crop_size[1]//2)
            ymax = ymin + self.crop_size[1]
            xmax = xmin + self.crop_size[0]
    
            return xmin,ymin,xmax,ymax
        
        if random.random()<self.prob:
            return generate_local_crop(mask,h,w)
        else:
            return generate_global_crop(h,w)
    
    def __call__(self, result):
        img = result['img']
        label = result['label']
        h,w = label.shape[:2]
        xmin,ymin,xmax,ymax = self.crop_bbox(label, h, w)
        result['img'] = img[ymin:ymax,xmin:xmax]
        result['label'] = label[ymin:ymax,xmin:xmax]
        return result
    
@TRANSFORMS.register_module
class RandomBlur(object):
    """
    Blurring an image by a Gaussian function with a certain probability.
    """

    def __init__(self, prob=0.1, blur_type="gaussian"):
        self.prob = prob
        self.blur_type = blur_type

    def __call__(self, result):
        img = result['img']
        if np.random.random() <= self.prob:
            radius = np.random.randint(3, 10)
            if radius % 2 != 1:
                radius = radius + 1
            if radius > 9:
                radius = 9
            if self.blur_type == "gaussian":
                result['img'] = cv2.GaussianBlur(img,(radius, radius), 0, 0)
            elif self.blur_type == "median":
                result['img'] = cv2.medianBlur(img, radius)
            elif self.blur_type == "blur":
                result['img'] = cv2.blur(img, (radius, radius))
            
        return result


@TRANSFORMS.register_module   
class SmartCrop(object):
    def __init__(self,prob=0.9):
        super().__init__()
        self.prob = prob
        self.crop_size = 256
        self.shift = 10

    def crop_bbox(self, mask, h, w, cls_id):
        def generate_global_crop(h,w):
            margin_h = max(h - self.crop_size, 0)
            margin_w = max(w - self.crop_size, 0)
            ymin = np.random.randint(0, margin_h + 1)
            xmin = np.random.randint(0, margin_w + 1)
            ymax = ymin + self.crop_size
            xmax = xmin + self.crop_size

            return xmin,ymin,xmax,ymax
        
        def generate_local_crop(mask,h,w,cls_id):
            if cls_id == 0:
                return generate_global_crop(h,w)
            m = np.where(mask==cls_id,1,0).astype(np.uint8)
            contours,_ = cv2.findContours(m,mode=cv2.RETR_LIST,method=cv2.CHAIN_APPROX_SIMPLE)  
            contour = random.choice(contours)
            rec = cv2.boundingRect(contour)
            cx = rec[0]+rec[2]//2 + random.randint(-self.shift,self.shift)
            cy = rec[1]+rec[3]//2 + random.randint(-self.shift,self.shift)

            xmin = min(w-self.crop_size,cx-self.crop_size//2) if cx > w//2 else max(0,cx-self.crop_size//2)
            ymin = min(h-self.crop_size,cy-self.crop_size//2) if cy > h//2 else max(0,cy-self.crop_size//2)

            ymax = ymin + self.crop_size
            xmax = xmin + self.crop_size
    
            return xmin,ymin,xmax,ymax
        
        if random.random()<self.prob:
            return generate_local_crop(mask,h,w,cls_id)
        else:
            return generate_global_crop(h,w)
    
    def __call__(self, result):
        img = result['img']
        label = result['label']
        cls_id = result['cls_id']
        self.crop_size = result['crop_size']
        self.shift = self.crop_size//25
        h,w = label.shape[:2]
        pad_height = max(self.crop_size-h, 0)
        pad_width = max(self.crop_size-w, 0) 
        if (pad_height > 0 or pad_width > 0):
            img = cv2.copyMakeBorder(img, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=128) #top,bottom,left,right
            label = cv2.copyMakeBorder(label, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=255)
            
        h,w = label.shape[:2]
        xmin,ymin,xmax,ymax = self.crop_bbox(label, h, w, cls_id)
        result['img'] = img[ymin:ymax,xmin:xmax]
        result['label'] = label[ymin:ymax,xmin:xmax]
        return result