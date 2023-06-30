import os
import cv2
import shutil
import random
from . import label_pb2 
import numpy as np
from tqdm import tqdm
import pandas as pd
from pycocotools.coco import COCO

Label = label_pb2.Label()

# 将阿丘的aqlabel标注转换为灰度mask标注
def aqlabel2mask(label_path,save_path,class_names):
    print('=============开始转换=====================')
    os.makedirs(save_path,exist_ok=True)
    cls_ids = {class_name:i+1 for i,class_name in enumerate(class_names)}
    for file in tqdm(os.scandir(label_path)):
        with open(file.path, "rb") as f:
            label_bin = f.read()
            lb = Label.FromString(label_bin)
            width = int(lb.img_size.width)
            height = int(lb.img_size.height)
            mask = np.zeros((height,width),dtype=np.uint8)
            for r in lb.regions:
                id_ = cls_ids[r.name]
                points = []
                for p in r.polygon.outer.points:
                    points.append([p.x, p.y])
                polygon = np.array(points,np.int32).reshape(-1,2)
                cv2.fillConvexPoly(mask, polygon, id_)
            save_p = os.path.join(save_path,file.name.replace('.aqlabel','.png'))
            cv2.imwrite(save_p,mask)
    print('=============转换结束=====================')

# 将灰度mask转换为阿丘的aqlabel格式
def mask2aqlabel(mask_path,save_path,class_names):
    os.makedirs(save_path,exist_ok=True)
    for i,f in tqdm(enumerate(os.scandir(mask_path))):
        filename = os.path.splitext(f.name)[0]+'.aqlabel'
        img = cv2.imread(f.path,0)
        height,width = img.shape[:2]
        class_ids = np.unique(img)

        label = label_pb2.Label()
        label.dataset_type = label.DataSetType.Segment
        label.img_size.width = width
        label.img_size.height = height

        for class_id in class_ids:
            if class_id == 0:
                continue
            mask = np.zeros((height,width),dtype=np.uint8)
            mask[img==class_id] = 1
            class_name = class_names[class_id-1]
            contours,_ = cv2.findContours(mask,mode=cv2.RETR_LIST,method=cv2.CHAIN_APPROX_SIMPLE) # 最简情况，只有外轮廓
            for contour in contours:
                ring = label_pb2.Ring()
                
                for point in contour.reshape(-1,2):
                    p = label_pb2.Point2f()
                    p.x = point[0]
                    p.y = point[1]
                    ring.points.extend([p])
                
                polygon = label_pb2.Polygon()
                polygon.outer.CopyFrom(ring)
                region = label_pb2.Region()
                region.name = class_name
                region.score = 1.0
                region.polygon.CopyFrom(polygon)
                label.regions.extend([region])
        with open(os.path.join(save_path, filename), "wb") as fp:
            fp.write(label.SerializeToString())
            
# 将coco的json标注转换为mask灰度掩码图像
def coco2mask(ann_path,save_path):
    os.makedirs(save_path,exist_ok=True)
    coco=COCO(ann_path)
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    for imgId in tqdm(imgIds):
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        mask = np.zeros((int(img['height']),int(img['width'])),np.uint8)
        if len(annIds) > 0:
            mask = coco.annToMask(anns[0]) * (anns[0]['category_id']+1)
            for i in range(len(anns) - 1):
                mask += coco.annToMask(anns[i + 1]) * (anns[i + 1]['category_id']+1)
        save_p = os.path.join(save_path,os.path.splitext(img['file_name'])[0]+'.png')
        cv2.imwrite(save_p,mask)

def get_ext(path):
    if os.path.isdir(path):
        for f in os.scandir(path):
            return get_ext(f.path)
    else:
        return os.path.splitext(path)[1]

def evaluate(result,class_names):
    result = tuple(zip(*result))
    result_intersect = sum(result[0])
    result_union = sum(result[1])
    result_pred_label = sum(result[2])
    result_label = sum(result[3])
 
    headers = []
    data = []
    metrics = []
    avg = ['average']

    headers.append('Class')
    headers.append('IoU')
    iou = result_intersect / (result_union+1)
    data.append(list(iou))
    avg.append(iou.mean())
    
    headers.append('Recall')
    acc = result_intersect / (result_label+1)
    data.append(list(acc))
    avg.append(acc.mean())

    headers.append('Precision')
    precision = result_intersect / (result_pred_label+1)
    data.append(list(precision))
    avg.append(precision.mean())
    
    for i,line in enumerate(zip(*data)):
        metrics.append([class_names[i],]+list(line)) 
    metrics.append(avg)
    df = pd.DataFrame(metrics,columns=headers)
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
      
def data_split(data_path='.',num_classes=2,label_prefix='labels',img_prefix='images',r=0.8):

    img_ext = get_ext(os.path.join(data_path,img_prefix))
    stat = {id_:{"area":[],'filename':[]} for id_ in range(1,num_classes+1)}
    stat[0] = {"area":[],'filename':[]}
    filenames = os.listdir(os.path.join(data_path,img_prefix))

    train = open(os.path.join(data_path,'train.txt'),'w')
    val = open(os.path.join(data_path,'val.txt'),'w')
    test = open(os.path.join(data_path,'test.txt'),'w')

    for f in filenames:
        basename = os.path.splitext(f)[0]
        mask_path = os.path.join(data_path,label_prefix,basename+'.png')
        line = os.path.join(img_prefix,basename+img_ext)+'\n'
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path)
            ids = list(np.unique(mask))
            del ids[0]
            if len(ids) == 0:
                test.write(line)
            for id_ in ids:
                area = (mask==id_).sum()
                if area>0:
                    stat[id_]["area"].append(area)
                    if len(ids) == 1:
                        stat[id_]["filename"].append(basename)
                    elif len(ids) > 1:
                        stat[0]["filename"].append(basename)
        else:
            test.write(line)
            
    for id_ in range(num_classes+1):
        filenames = list(set(stat[id_]['filename']))
        num = len(filenames)
        x += num
        n = int(num*r+0.5)
        random.shuffle(filenames)
        if id_>0:
            stat[id_]['area'] = sum(stat[id_]['area'])/len(stat[id_]['area'])
        for i in range(num):
            basename = filenames[i]
            line = os.path.join(img_prefix,basename+img_ext)+' '+ os.path.join(label_prefix,basename+'.png')+'\n'
            if i<n:
                train.write(line)
            else:
                val.write(line)
    train.close()
    val.close()
    test.close()
    print("====================================")
    print("Dataset Information:",stat)
    print("====================================")
    
def move_img(data_path,txt_name='train.txt',save_path='aqrose',img_dir='source',lb_dir='label'):
    txt_path = os.path.join(data_path,txt_name)
    save_img = os.path.join(data_path,save_path,img_dir)
    save_lb = os.path.join(data_path,save_path,lb_dir)
    if os.path.exists(save_img):
        os.remove(save_img)
    if os.path.exists(save_lb):
        os.remove(save_lb)
    os.makedirs(save_img,exist_ok=True)
    os.makedirs(save_lb,exist_ok=True)
    for line in open(txt_path).readlines():
        item = line.strip().split()
        #item[1] = item[1].replace('labels','aqlabel').replace('.png','.aqlabel')
        img_p = os.path.join(data_path,item[0])
        shutil.copy(img_p,save_img+'/'+os.path.basename(item[0]))
        if len(item)>1:
            lb_p = os.path.join(data_path,item[1])
            shutil.copy(lb_p,save_lb+'/'+os.path.basename(item[1]))
        

# 优化的随机裁剪策略，增强前景裁剪概率
def random_crop(mask_path,img_path,prob=0.7,crop_size=(512,512),shift=50):
    img = cv2.imread(mask_path,0)
    h,w = img.shape[:2]
    
    cv2.namedWindow('img',0)
    image = cv2.imread(img_path)
    
    def global_crop(h,w):
        margin_h = max(h - crop_size[0], 0)
        margin_w = max(w - crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        ymin, ymax = offset_h, offset_h + crop_size[0]
        xmin, xmax = offset_w, offset_w + crop_size[1]
        return xmin,ymin,xmax,ymax
        
    def local_crop(h,w):
        cls_id = list(np.unique(img))[1:]
        random_id = random.choice(cls_id)
        loc = np.where(img==random_id)
        random_index = random.choice(range(loc[0].shape[0]))
        x = loc[1][random_index]+random.randint(-shift,shift)
        y = loc[0][random_index]+random.randint(-shift,shift)
        xmin = min(w-crop_size[1],x-crop_size[1]//2) if x > w//2 else max(0,x-crop_size[1]//2)
        ymin = min(h-crop_size[0],y-crop_size[0]//2) if y > h//2 else max(0,y-crop_size[0]//2)
        xmax = xmin+crop_size[1]
        ymax = ymin+crop_size[0]
        return xmin,ymin,xmax,ymax
    if random.random()<prob:
        xmin,ymin,xmax,ymax = local_crop(img,h,w)
    else:
        xmin,ymin,xmax,ymax = global_crop(h,w)
    cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(255,0,0),2)
    cv2.imshow('img',image)
    cv2.waitKey(0)


