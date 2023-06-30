import argparse
from tools import SegmentationMetric

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--pred_label_path', default=None, help='model path')
    parser.add_argument('--label_path', default=None, help='data path')
    parser.add_argument('--img_path', default=None, help='data path')
    parser.add_argument('--save_path', default=None, help='data path')
    args = parser.parse_args()

    class_names = ['zw','posun','H1','B3','B2','E1','E2'] # no background
    metric = SegmentationMetric(args.pred_label_path,args.label_path,args.img_path,args.save_path,class_names,img_ext='.png')
    metric.compute_metrics()





