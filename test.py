import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from apis.infer import SegmentModel
from tools import evaluate,intersect_and_union


# example for infer api call.
def infer(args):
    os.makedirs(args.save_path,exist_ok=True)
    
    runtime = SegmentModel(args.config_path)
    runtime.set_model_path(args.model_path)
    runtime.load_model()
    class_names = runtime.class_names
    num_classes = runtime.num_classes
    anno_path = os.path.join(args.data_path,args.anno_file)
    lines = open(anno_path).readlines()
    results = []
    

    for line in tqdm(lines):
        item = line.strip().split()
        img_path = os.path.join(args.data_path,item[0])
        lb_path = os.path.join(args.data_path,item[1])

        image = cv2.imread(img_path)
        h,w = image.shape[:2]
        label = np.zeros((h,w),np.uint8)
        if os.path.exists(lb_path):
            label = cv2.imread(lb_path,0)

        result = runtime.predict(image)

        iou = intersect_and_union(result,label,num_classes)
        results.append(iou)
        if not args.no_render:
            result = runtime.pre_to_img(image,result)
        sp = os.path.join(args.save_path,os.path.basename(item[0]))
        cv2.imwrite(sp,result)
    evaluate(results,class_names)

def parse_args():

    parser = argparse.ArgumentParser()
    dataset = 'wafer'
    parser.add_argument('--config_path', default=f'work_dirs/{dataset}/crop_blob/config.yaml') 
    parser.add_argument('--model_path', default=None) 
    parser.add_argument('--input_size', default=None,help='w,h')
    parser.add_argument('--data_path', default=f'../../dataset/{dataset}')
    parser.add_argument('--anno_file', default='val.txt')
    parser.add_argument('--save_path', default=f'work_dirs/{dataset}/crop_blob/result')
    parser.add_argument('--no_render', action='store_true', default=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    infer(args)